#include "application.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

#include <imgui.h>

#ifndef SHADER_DIRECTORY
#define SHADER_DIRECTORY "shaders"
#endif

namespace application {

namespace {

using namespace math;

// 6 граней по 4 вершины, у каждой грани свои вершины, чтобы у неё была своя нормаль
constexpr uint32_t vertex_count = 24;
// 6 граней по 2 треугольника по 3 индекса
constexpr uint32_t index_count = 36;

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
};

// Параллелепипед задаётся длинами трёх рёбер и углами между ними
struct ParallelepipedShape {
	float a = 2.0f; // длины рёбер
	float b = 1.2f;
	float c = 0.8f;
	float alpha = 90.0f; // угол между b и c в градусах
	float beta = 90.0f; // угол между a и c
	float gamma = 90.0f; // угол между a и b
};

enum Projection : int {
	projection_perspective,
	projection_orthographic,
};

enum Trajectory : int {
	trajectory_wavy_circle,
	trajectory_spiral,
	trajectory_figure_eight,
};

// Всё, что меняется из интерфейса
struct State {
	// Фигура (вариант 3)
	ParallelepipedShape shape;
	Vec3 edges[3];
	bool shape_valid = true;
	bool geometry_dirty = false;

	// Задание 1: проекция и камера
	int projection = projection_perspective;
	float fov = 60.0f; // угол обзора по вертикали в градусах
	float ortho_height = 9.2f; // высота видимой области для ортографической проекции
	float near_plane = 0.1f;
	float far_plane = 100.0f;
	float camera_distance = 8.0f;
	float camera_yaw = 30.0f; // азимут в градусах
	float camera_pitch = 25.0f; // наклон в градусах

	// Задание 2: положение, поворот и растяжение
	Vec3 position = { 0.0f, 0.0f, 0.0f };
	Vec3 rotation = { 0.0f, 0.0f, 0.0f }; // углы в градусах
	Vec3 scale = { 1.0f, 1.0f, 1.0f };

	// Задание 3: анимация
	bool playing = true;
	float animation_speed = 1.0f; // множитель скорости
	float animation_time = 0.0f; // время анимации в секундах
	int trajectory = trajectory_wavy_circle;
	float orbit_radius = 2.5f;
	float orbit_speed = 45.0f; // градусов в секунду по траектории
	float height_amplitude = 0.8f;
	float wave_count = 2.0f; // сколько волн по высоте за один оборот
	float spiral_turns = 3.0f; // сколько витков спираль делает за один подъём
	float spin_speed = 90.0f; // градусов в секунду вокруг своей оси
	int spin_axis = 1; // 0 это X, 1 это Y, 2 это Z

	// Задания 4 и 5: цвет
	float color[3] = { 1.0f, 1.0f, 1.0f };
	bool use_vertex_colors = true;
	bool use_lighting = false;

	double previous_time = -1.0;
};

State state;
UniformData uniform_data;

Buffer vertex_buffer;
Buffer index_buffer;
Buffer uniform_buffer;

VkDescriptorSetLayout vk_descriptor_set_layout;
VkDescriptorPool vk_descriptor_pool;
VkDescriptorSet vk_descriptor_set;

VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_pipeline;

// Считает векторы рёбер по длинам и углам.
// Ребро a лежит на оси X, ребро b в плоскости XY, ребро c смотрит в сторону +Z.
// Возвращает false, если такие три угла не могут сойтись в одной вершине.
bool computeEdges(const ParallelepipedShape& shape, Vec3 edges[3]) {
	const float cos_alpha = std::cos(radians(shape.alpha));
	const float cos_beta = std::cos(radians(shape.beta));
	const float cos_gamma = std::cos(radians(shape.gamma));
	const float sin_gamma = std::sin(radians(shape.gamma));

	// Направление c находим из того, что угол с a равен beta, а угол с b равен alpha
	const float cx = cos_beta;
	const float cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
	const float cz_squared = 1.0f - cx * cx - cy * cy;
	if (cz_squared < 0.01f) {
		return false;
	}

	edges[0] = Vec3{ 1.0f, 0.0f, 0.0f } * shape.a;
	edges[1] = Vec3{ cos_gamma, sin_gamma, 0.0f } * shape.b;
	edges[2] = Vec3{ cx, cy, std::sqrt(cz_squared) } * shape.c;
	return true;
}

// Цвет вершины по её координатам: x даёт красный, y зелёный, z синий.
// Каждая координата переводится в число от 0.15 до 1.
Vec3 proceduralColor(Vec3 p, Vec3 half_extent) {
	auto channel = [](float x, float half) {
		return 0.15f + 0.85f * (0.5f + 0.5f * x / half);
	};
	return { channel(p.x, half_extent.x), channel(p.y, half_extent.y), channel(p.z, half_extent.z) };
}

// Заполняет 24 вершины и 36 индексов параллелепипеда с центром в начале координат.
// Вершины каждого треугольника идут против часовой стрелки, если смотреть на грань снаружи.
void generateParallelepiped(const Vec3 edges[3], Vertex* vertices, uint16_t* indices) {
	// Половина размеров фигуры по осям, нужна для расчёта цвета
	const Vec3 half_extent = Vec3{
		std::fabs(edges[0].x) + std::fabs(edges[1].x) + std::fabs(edges[2].x),
		std::fabs(edges[0].y) + std::fabs(edges[1].y) + std::fabs(edges[2].y),
		std::fabs(edges[0].z) + std::fabs(edges[1].z) + std::fabs(edges[2].z),
	} * 0.5f;

	uint32_t vertex = 0;
	uint32_t index = 0;

	// Для каждого ребра две противоположные грани, каждая натянута на два других ребра
	for (int axis = 0; axis < 3; ++axis) {
		for (float side : { -0.5f, 0.5f }) {
			const Vec3 center = edges[axis] * side;
			Vec3 u = edges[(axis + 1) % 3];
			Vec3 w = edges[(axis + 2) % 3];
			Vec3 normal = normalize(cross(u, w));

			// Нормаль должна смотреть наружу. Если она смотрит внутрь, меняем u и w местами,
			// тогда нормаль разворачивается и порядок обхода вершин тоже.
			if (dot(normal, center) < 0.0f) {
				std::swap(u, w);
				normal = -normal;
			}

			const Vec3 corners[4] = {
				center - u * 0.5f - w * 0.5f,
				center + u * 0.5f - w * 0.5f,
				center + u * 0.5f + w * 0.5f,
				center - u * 0.5f + w * 0.5f,
			};

			const uint16_t first = static_cast<uint16_t>(vertex);
			for (const Vec3& corner : corners) {
				vertices[vertex++] = { corner, normal, proceduralColor(corner, half_extent) };
			}

			// Грань из вершин 0, 1, 2, 3 делится на треугольники 0-1-2 и 0-2-3
			const uint16_t quad[6] = { 0, 1, 2, 0, 2, 3 };
			for (uint16_t k : quad) {
				indices[index++] = static_cast<uint16_t>(first + k);
			}
		}
	}
}

// Создаёт буфер в памяти, доступной процессору. Память сразу отображается и остаётся отображённой.
bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& buffer) {
	auto& context = graphics::internal::context;

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo allocation_result = {};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &buffer.buffer, &buffer.allocation, &allocation_result) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan buffer\n";
		return false;
	}

	buffer.mapped = allocation_result.pMappedData;
	return true;
}

void writeBuffer(const Buffer& buffer, const void* data, VkDeviceSize size) {
	std::memcpy(buffer.mapped, data, size_t(size));
	// Нужно, если память не HOST_COHERENT, иначе ничего не делает
	vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, 0, size);
}

void destroyBuffer(Buffer& buffer) {
	vmaDestroyBuffer(graphics::internal::context.allocator, buffer.buffer, buffer.allocation);
	buffer = {};
}

void uploadGeometry() {
	Vertex vertices[vertex_count];
	uint16_t indices[index_count];
	generateParallelepiped(state.edges, vertices, indices);
	writeBuffer(vertex_buffer, vertices, sizeof(vertices));
	writeBuffer(index_buffer, indices, sizeof(indices));
}

VkShaderModule loadShaderModule(const char* file_name) {
	const std::filesystem::path path = std::filesystem::path(SHADER_DIRECTORY) / file_name;

	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader file " << path << '\n';
		return VK_NULL_HANDLE;
	}

	const std::streamsize size = file.tellg();
	// SPIR-V состоит из 32-битных слов, поэтому читаем файл в массив uint32_t
	std::vector<uint32_t> code(size_t(size) / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), size);

	const VkShaderModuleCreateInfo shader_module_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size() * sizeof(uint32_t),
		.pCode = code.data(),
	};

	VkShaderModule shader_module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &shader_module_info,
	                         nullptr, &shader_module) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan shader module from " << path << '\n';
		return VK_NULL_HANDLE;
	}

	return shader_module;
}

bool createDescriptors() {
	auto& context = graphics::internal::context;

	// binding 0 это uniform-буфер, его читают оба шейдера
	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo set_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &set_layout_info, nullptr,
	                                &vk_descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr,
	                           &vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor pool\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vk_descriptor_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &set_info, &vk_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan descriptor set\n";
		return false;
	}

	// Привязываем uniform-буфер к binding 0
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = uniform_buffer.buffer,
		.offset = 0,
		.range = sizeof(UniformData),
	};

	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vk_descriptor_set,
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
	return true;
}

bool createPipeline() {
	auto& context = graphics::internal::context;

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &vk_descriptor_set_layout,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr,
	                           &vk_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		return false;
	}

	const VkShaderModule vertex_shader = loadShaderModule("parallelepiped.vert.spv");
	const VkShaderModule fragment_shader = loadShaderModule("parallelepiped.frag.spv");
	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	};

	// Описание вершины, должно совпадать со struct Vertex и входами вершинного шейдера
	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		},
		{
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, color),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = sizeof(vertex_attributes) / sizeof(vertex_attributes[0]),
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE,
	};

	// Viewport и scissor задаются в render(), поэтому при изменении размера окна конвейер не пересоздаётся
	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	// Снаружи треугольники идут против часовой стрелки. Переворот Y в матрице проекции
	// сохраняет этот порядок на экране, поэтому задние грани можно отсекать.
	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	// Буфер глубины создаётся в стартовом коде и каждый кадр заполняется значением 1.0
	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = sizeof(stages) / sizeof(stages[0]),
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = vk_pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                  &pipeline_info, nullptr, &vk_pipeline);

	// После создания конвейера шейдерные модули больше не нужны
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan graphics pipeline\n";
		return false;
	}

	return true;
}

bool createResources() {
	if (!computeEdges(state.shape, state.edges)) {
		std::cerr << "Default parallelepiped shape is invalid\n";
		return false;
	}

	if (!createBuffer(sizeof(Vertex) * vertex_count, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) ||
	    !createBuffer(sizeof(uint16_t) * index_count, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer) ||
	    !createBuffer(sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, uniform_buffer)) {
		return false;
	}

	uploadGeometry();

	return createDescriptors() && createPipeline();
}

// Смещение фигуры по траектории в момент времени t
Vec3 trajectoryOffset(float t) {
	const float theta = radians(state.orbit_speed) * t; // угол на траектории
	const float r = state.orbit_radius;
	const float h = state.height_amplitude;

	switch (state.trajectory) {
	case trajectory_spiral: {
		// Спираль: за spiral_turns оборотов поднимается от -h до h, потом опускается обратно
		const float progress = theta / (2.0f * pi * state.spiral_turns);
		float phase = std::fmod(progress, 2.0f);
		if (phase < 0.0f) {
			phase += 2.0f;
		}
		const float climb = phase <= 1.0f ? phase : 2.0f - phase; // растёт от 0 до 1 и падает обратно до 0
		return { r * std::cos(theta), h * (2.0f * climb - 1.0f), r * std::sin(theta) };
	}

	case trajectory_figure_eight:
		// Восьмёрка (лемниската Жероно) в плоскости XZ и волны по высоте
		return {
			r * std::sin(theta),
			h * std::sin(state.wave_count * theta),
			r * std::sin(theta) * std::cos(theta),
		};

	case trajectory_wavy_circle:
	default:
		// Окружность в плоскости XZ, высота меняется wave_count раз за оборот
		return {
			r * std::cos(theta),
			h * std::sin(state.wave_count * theta),
			r * std::sin(theta),
		};
	}
}

// Поворот вокруг своей оси, ось выбирается в интерфейсе
Mat4 spinMatrix(float angle) {
	switch (state.spin_axis) {
	case 0: return rotateX(angle);
	case 2: return rotateZ(angle);
	default: return rotateY(angle);
	}
}

// Считает матрицы и цвет для uniform-буфера
void updateUniformData() {
	const VkExtent2D extent = graphics::internal::context.swapchain_extent;
	const float aspect = extent.height > 0 ? float(extent.width) / float(extent.height) : 1.0f;

	// Матрица модели: сначала растяжение, потом вращение вокруг своей оси,
	// потом поворот из интерфейса и перенос (позиция плюс смещение по траектории)
	const Vec3 rotation_radians = {
		radians(state.rotation.x), radians(state.rotation.y), radians(state.rotation.z),
	};
	const Mat4 rotation = rotateEuler(rotation_radians) *
	                      spinMatrix(radians(state.spin_speed) * state.animation_time);
	const Vec3 offset = trajectoryOffset(state.animation_time);
	const Mat4 model = translate(state.position + offset) * rotation * scale(state.scale);

	// Матрица для нормалей: поворот, умноженный на обратное растяжение.
	// Так нормали остаются перпендикулярными граням даже при неравномерном растяжении.
	const Mat4 normal_matrix = rotation * scale({
		1.0f / state.scale.x, 1.0f / state.scale.y, 1.0f / state.scale.z,
	});

	// Матрица вида: камера двигается по сфере вокруг начала координат
	const float yaw = radians(state.camera_yaw);
	const float pitch = radians(state.camera_pitch);
	const Vec3 eye = Vec3{
		std::cos(pitch) * std::sin(yaw),
		std::sin(pitch),
		std::cos(pitch) * std::cos(yaw),
	} * state.camera_distance;
	const Mat4 view = lookAt(eye, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });

	// Матрица проекции (задание 1)
	Mat4 projection;
	if (state.projection == projection_perspective) {
		projection = perspective(radians(state.fov), aspect, state.near_plane, state.far_plane);
	} else {
		const float half_height = state.ortho_height * 0.5f;
		const float half_width = half_height * aspect;
		projection = orthographic(-half_width, half_width, -half_height, half_height,
		                          state.near_plane, state.far_plane);
	}

	uniform_data.mvp = projection * view * model;
	uniform_data.normal_matrix = normal_matrix;
	uniform_data.color = { state.color[0], state.color[1], state.color[2], 1.0f };
	uniform_data.use_vertex_colors = state.use_vertex_colors ? 1u : 0u;
	uniform_data.use_lighting = state.use_lighting ? 1u : 0u;
}

// Панель ImGui со всеми настройками
void drawInterface() {
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(420.0f, 700.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Лабораторная работа №1");

	ImGui::Text("Вариант 3: параллелепипед. FPS: %.0f", ImGui::GetIO().Framerate);

	if (ImGui::CollapsingHeader("Фигура", ImGuiTreeNodeFlags_DefaultOpen)) {
		ParallelepipedShape& shape = state.shape;
		bool changed = false;
		changed |= ImGui::SliderFloat("Ребро a", &shape.a, 0.2f, 4.0f, "%.2f");
		changed |= ImGui::SliderFloat("Ребро b", &shape.b, 0.2f, 4.0f, "%.2f");
		changed |= ImGui::SliderFloat("Ребро c", &shape.c, 0.2f, 4.0f, "%.2f");
		changed |= ImGui::SliderFloat("Угол α (b, c)", &shape.alpha, 30.0f, 150.0f, "%.0f°");
		changed |= ImGui::SliderFloat("Угол β (a, c)", &shape.beta, 30.0f, 150.0f, "%.0f°");
		changed |= ImGui::SliderFloat("Угол γ (a, b)", &shape.gamma, 30.0f, 150.0f, "%.0f°");
		if (ImGui::Button("Прямоугольный")) {
			shape.alpha = shape.beta = shape.gamma = 90.0f;
			changed = true;
		}

		// Если форма изменилась, пересчитываем рёбра. Вершинный буфер обновится в render().
		if (changed) {
			Vec3 edges[3];
			state.shape_valid = computeEdges(shape, edges);
			if (state.shape_valid) {
				std::memcpy(state.edges, edges, sizeof(edges));
				state.geometry_dirty = true;
			}
		}

		if (!state.shape_valid) {
			ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
			                   "Такие углы не образуют параллелепипед");
		}
	}

	// Задание 1
	if (ImGui::CollapsingHeader("1. Проекция и камера", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::RadioButton("Перспективная", &state.projection, projection_perspective);
		ImGui::SameLine();
		ImGui::RadioButton("Ортографическая", &state.projection, projection_orthographic);

		if (state.projection == projection_perspective) {
			ImGui::SliderFloat("Угол обзора", &state.fov, 20.0f, 120.0f, "%.0f°");
		} else {
			ImGui::SliderFloat("Высота области", &state.ortho_height, 1.0f, 40.0f, "%.1f");
			if (ImGui::Button("Как у перспективы")) {
				// Подбираем высоту так, чтобы фигура в центре была такого же размера, как в перспективе
				state.ortho_height = 2.0f * state.camera_distance *
				                     std::tan(radians(state.fov) * 0.5f);
			}
		}

		ImGui::SliderFloat("Ближняя плоскость", &state.near_plane, 0.01f, 5.0f, "%.2f");
		ImGui::SliderFloat("Дальняя плоскость", &state.far_plane, 10.0f, 200.0f, "%.0f");
		ImGui::SliderFloat("Расстояние камеры", &state.camera_distance, 3.0f, 40.0f, "%.1f");
		ImGui::SliderFloat("Азимут камеры", &state.camera_yaw, -180.0f, 180.0f, "%.0f°");
		ImGui::SliderFloat("Наклон камеры", &state.camera_pitch, -89.0f, 89.0f, "%.0f°");
	}

	// Задание 2
	if (ImGui::CollapsingHeader("2. Положение, поворот, растяжение", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Позиция", &state.position.x, 0.02f, -10.0f, 10.0f, "%.2f");
		ImGui::SliderFloat3("Поворот X/Y/Z", &state.rotation.x, -180.0f, 180.0f, "%.0f°");
		// Растяжение не меньше 0.05 даже при вводе с клавиатуры.
		// При нуле или отрицательном значении ломается матрица нормалей и переворачиваются грани.
		ImGui::DragFloat3("Растяжение", &state.scale.x, 0.01f, 0.05f, 5.0f, "%.2f",
		                  ImGuiSliderFlags_AlwaysClamp);
		if (ImGui::Button("Сбросить трансформации")) {
			state.position = { 0.0f, 0.0f, 0.0f };
			state.rotation = { 0.0f, 0.0f, 0.0f };
			state.scale = { 1.0f, 1.0f, 1.0f };
		}
	}

	// Задание 3
	if (ImGui::CollapsingHeader("3. Анимация", ImGuiTreeNodeFlags_DefaultOpen)) {
		// Часть после ### это id кнопки, он не меняется, когда меняется надпись
		if (ImGui::Button(state.playing ? "Пауза###play" : "Воспроизвести###play")) {
			state.playing = !state.playing;
		}
		ImGui::SameLine();
		if (ImGui::Button("В начало")) {
			state.animation_time = 0.0f;
		}

		ImGui::SliderFloat("Скорость анимации", &state.animation_speed, 0.0f, 5.0f, "x%.2f");
		ImGui::Combo("Траектория", &state.trajectory,
		             "Окружность с волнами\0Спираль вверх-вниз\0Восьмёрка\0");
		ImGui::SliderFloat("Радиус", &state.orbit_radius, 0.0f, 6.0f, "%.2f");
		ImGui::SliderFloat("Скорость по траектории", &state.orbit_speed, -180.0f, 180.0f, "%.0f°/с");
		ImGui::SliderFloat("Высота", &state.height_amplitude, 0.0f, 4.0f, "%.2f");
		if (state.trajectory == trajectory_spiral) {
			ImGui::SliderFloat("Витков за подъём", &state.spiral_turns, 0.5f, 6.0f, "%.1f");
		} else {
			ImGui::SliderFloat("Волн за оборот", &state.wave_count, 0.0f, 8.0f, "%.1f");
		}
		ImGui::SliderFloat("Вращение вокруг оси", &state.spin_speed, -360.0f, 360.0f, "%.0f°/с");
		ImGui::Combo("Ось вращения", &state.spin_axis, "X\0Y\0Z\0");
	}

	// Задания 4 и 5
	if (ImGui::CollapsingHeader("4-5. Цвет", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Цвет фигуры", state.color);
		ImGui::Checkbox("Цвета вершин по координатам", &state.use_vertex_colors);
		ImGui::TextDisabled(state.use_vertex_colors ? "Итоговый цвет = цвет вершины * цвет из UI"
		                                            : "Итоговый цвет = цвет из UI");
		ImGui::Checkbox("Затенение граней по нормалям", &state.use_lighting);
	}

	ImGui::End();
}

} // namespace

bool initialize() {
	// В стандартном шрифте ImGui нет русских букв, поэтому загружаем системный шрифт
	const char* font_path = "C:/Windows/Fonts/segoeui.ttf";
	if (std::filesystem::exists(font_path)) {
		ImGui::GetIO().Fonts->AddFontFromFileTTF(font_path, 16.0f);
	}

	if (!createResources()) {
		shutdown();
		return false;
	}

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, vk_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr); // набор дескрипторов удаляется вместе с пулом
	vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout, nullptr);

	destroyBuffer(uniform_buffer);
	destroyBuffer(index_buffer);
	destroyBuffer(vertex_buffer);
}

void update(double time) {
	if (state.previous_time < 0.0) {
		state.previous_time = time;
	}
	float dt = float(time - state.previous_time);
	state.previous_time = time;
	if (dt > 0.1f) {
		dt = 0.1f; // чтобы фигура не прыгала после перетаскивания окна или остановки в отладчике
	}

	// На паузе время анимации не идёт
	if (state.playing) {
		state.animation_time += dt * state.animation_speed;
	}

	drawInterface();
	updateUniformData();
}

void render(const graphics::internal::FrameData& fd) {
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	auto& context = graphics::internal::context;

	// prepare() уже дождался конца прошлого кадра, значит GPU больше не читает эти буферы
	// и их можно перезаписать. В update() этого делать нельзя.
	if (state.geometry_dirty) {
		uploadGeometry();
		state.geometry_dirty = false;
	}
	writeBuffer(uniform_buffer, &uniform_data, sizeof(uniform_data));

	const VkCommandBuffer cmd = fd.command_buffer;
	vkResetCommandBuffer(cmd, 0);

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin_info);

	// Очистка: цвет фона и глубина 1.0
	const VkClearValue clear_values[] = {
		{ .color = { .float32 = { 0.02f, 0.025f, 0.035f, 1.0f } } },
		{ .depthStencil = { .depth = 1.0f, .stencil = 0 } },
	};

	const VkRenderPassBeginInfo render_pass_info = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .offset = { 0, 0 }, .extent = context.swapchain_extent },
		.clearValueCount = sizeof(clear_values) / sizeof(clear_values[0]),
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor = { .offset = { 0, 0 }, .extent = context.swapchain_extent };
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
	                        0, 1, &vk_descriptor_set, 0, nullptr);

	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &vertex_offset);
	vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);

	vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
