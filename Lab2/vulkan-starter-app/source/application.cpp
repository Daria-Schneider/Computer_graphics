#include "application.hpp"

#include <algorithm>
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

// Сколько источников помещается в storage-буфер после запуска.
// Это не предел: если источников станет больше, буфер пересоздаётся большего размера.
constexpr uint32_t initial_light_capacity = 8;

// Номера binding в наборе дескрипторов сцены (набор 0)
constexpr uint32_t scene_binding = 0;
constexpr uint32_t point_lights_binding = 1;
constexpr uint32_t spot_lights_binding = 2;

constexpr Vec3 world_up = { 0.0f, 1.0f, 0.0f };

// Визуализация источников (дополнительное задание 4): длина ребра кубика,
// число отрезков в окружности конуса прожектора и число его образующих
constexpr float marker_cube_size = 0.2f;
constexpr int cone_circle_segments = 32;
constexpr int cone_side_lines = 8;

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

// Параметры материала по модели Блинна-Фонга
struct Material {
	Vec3 albedo = { 0.8f, 0.8f, 0.8f }; // цвет рассеянного отражения
	Vec3 specular_color = { 0.5f, 0.5f, 0.5f }; // цвет блика
	float shininess = 32.0f; // коэффициент блика, чем он больше, тем меньше и резче блик
};

// Объект сцены: параллелепипед со своей формой, положением, материалом и буферами
struct SceneObject {
	const char* name = "";
	ParallelepipedShape shape;
	Vec3 edges[3] = {};
	bool shape_valid = true;
	bool geometry_dirty = false;

	Vec3 position = { 0.0f, 0.0f, 0.0f };
	Vec3 rotation = { 0.0f, 0.0f, 0.0f }; // углы в градусах
	Vec3 scale = { 1.0f, 1.0f, 1.0f };
	float spin_speed = 0.0f; // вращение вокруг вертикальной оси, градусов в секунду
	float spin_angle = 0.0f; // текущий угол этого вращения в градусах

	Material material;

	// Данные для uniform-буфера объекта, считаются в update()
	ModelData model_data = {};

	Buffer vertex_buffer;
	Buffer index_buffer;
	Buffer uniform_buffer;
	// Набор дескрипторов модели (набор 1), в нём uniform-буфер этого объекта
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

// Свободная камера: позиция и два угла поворота.
// По умолчанию камера сдвинута влево, чтобы сцена была видна справа от панели интерфейса.
struct Camera {
	Vec3 position = { -3.2f, 3.8f, 9.0f };
	float yaw = 0.0f; // поворот вокруг вертикали в градусах, при 0 камера смотрит вдоль -Z
	float pitch = -18.0f; // наклон вверх или вниз в градусах
	float fov = 60.0f; // угол обзора по вертикали в градусах
	float near_plane = 0.1f;
	float far_plane = 100.0f;
	float speed = 4.0f; // скорость движения с клавиатуры, единиц в секунду
};

struct AmbientLight {
	Vec3 color = { 0.6f, 0.7f, 1.0f };
	float intensity = 0.1f;
};

// Направление источников задаётся теми же двумя углами, что и у камеры
struct DirectionalLight {
	bool enabled = true;
	Vec3 color = { 1.0f, 0.95f, 0.85f };
	float intensity = 0.25f;
	float yaw = 40.0f; // куда светит источник, градусы
	float pitch = -50.0f;
};

struct PointLight {
	bool enabled = true;
	Vec3 position = { 0.0f, 2.0f, 2.0f };
	Vec3 color = { 1.0f, 1.0f, 1.0f };
	float intensity = 6.0f;
};

struct SpotLight {
	bool enabled = true;
	bool attached_to_camera = false; // фонарик: позиция и направление берутся у камеры
	Vec3 position = { 0.0f, 4.0f, 3.0f };
	float yaw = 0.0f; // куда светит прожектор, градусы
	float pitch = -60.0f;
	float inner_angle = 15.0f; // угол от оси до края полностью освещённой части, градусы
	float outer_angle = 25.0f; // угол от оси до границы освещённой области, градусы
	Vec3 color = { 1.0f, 1.0f, 1.0f };
	float intensity = 15.0f;
};

// Буфер, размер данных в котором меняется во время работы: storage-буфер с источниками
// или вершинный буфер визуализации. capacity это сколько элементов в него помещается.
// Если данных становится больше, буфер пересоздаётся большего размера.
struct GrowingBuffer {
	Buffer buffer;
	uint32_t capacity = 0;
};

// Всё, что меняется из интерфейса
struct State {
	Camera camera;
	bool camera_rotating = false; // правая кнопка мыши зажата над сценой

	AmbientLight ambient;
	DirectionalLight directional;
	std::vector<PointLight> point_lights;
	std::vector<SpotLight> spot_lights;

	std::vector<SceneObject> objects;
	int selected_object = 0;

	// Задание 4: визуализация источников
	bool show_light_cubes = true;
	bool show_spot_cones = true;
	float cone_length = 1.5f; // длина образующих конуса прожектора

	double previous_time = -1.0;
};

State state;

// Данные для буферов сцены, считаются в update(), записываются в render()
SceneData scene_data;
std::vector<PointLightData> point_light_data;
std::vector<SpotLightData> spot_light_data;

// Вершины визуализации источников, строятся в update(): сначала треугольники кубиков, потом отрезки конусов
std::vector<MarkerVertex> marker_vertices;
uint32_t marker_triangle_vertex_count = 0;
// Кубик источника: 36 вершин треугольников вокруг начала координат, строится один раз при запуске
Vec3 marker_cube[index_count];

Buffer scene_buffer;
GrowingBuffer point_light_buffer;
GrowingBuffer spot_light_buffer;
GrowingBuffer marker_buffer;

VkDescriptorSetLayout vk_scene_set_layout;
VkDescriptorSetLayout vk_model_set_layout;
VkDescriptorPool vk_descriptor_pool;
VkDescriptorSet vk_scene_set;

// У всех конвейеров одна раскладка: набор сцены и набор модели
VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_lit_pipeline; // объекты с освещением по Блинну-Фонгу
VkPipeline vk_marker_pipeline; // кубики источников, без освещения
VkPipeline vk_marker_line_pipeline; // конусы прожекторов отрезками, без освещения

// Единичный вектор направления по двум углам в градусах.
// yaw это поворот вокруг вертикали: при 0 вектор смотрит вдоль -Z, при 90 вдоль +X.
// pitch это наклон: положительный вверх, отрицательный вниз.
Vec3 directionFromAngles(float yaw, float pitch) {
	const float y = radians(yaw);
	const float p = radians(pitch);
	return { std::sin(y) * std::cos(p), std::sin(p), -std::cos(y) * std::cos(p) };
}

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

// Заполняет 24 вершины и 36 индексов параллелепипеда с центром в начале координат.
// Нормали считаются процедурно: нормаль грани это векторное произведение двух её рёбер.
// Вершины каждого треугольника идут против часовой стрелки, если смотреть на грань снаружи.
void generateParallelepiped(const Vec3 edges[3], Vertex* vertices, uint16_t* indices) {
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

			// У всех четырёх вершин грани одна и та же нормаль
			const uint16_t first = static_cast<uint16_t>(vertex);
			for (const Vec3& corner : corners) {
				vertices[vertex++] = { corner, normal };
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

void uploadGeometry(const SceneObject& object) {
	Vertex vertices[vertex_count];
	uint16_t indices[index_count];
	generateParallelepiped(object.edges, vertices, indices);
	writeBuffer(object.vertex_buffer, vertices, sizeof(vertices));
	writeBuffer(object.index_buffer, indices, sizeof(indices));
}

// Привязывает буфер целиком к binding в наборе дескрипторов
void writeBufferDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type, VkBuffer buffer) {
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE,
	};

	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = binding,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = type,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

// Storage-буфер на capacity источников, перед массивом место под количество
bool createLightBuffer(VkDeviceSize light_size, uint32_t capacity, Buffer& buffer) {
	return createBuffer(sizeof(LightBufferHeader) + light_size * capacity,
	                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buffer);
}

// Записывает источники в storage-буфер: сначала их количество, потом массив.
// Если источников больше, чем помещается, буфер пересоздаётся в два раза больше.
// Вызывается только из render(), когда GPU уже закончил прошлый кадр и не читает старый буфер.
template <typename LightData>
void uploadLights(GrowingBuffer& light_buffer, uint32_t binding, const std::vector<LightData>& lights) {
	uint32_t count = uint32_t(lights.size());

	if (count > light_buffer.capacity) {
		const uint32_t capacity = std::max(count, light_buffer.capacity * 2);
		Buffer new_buffer;
		if (createLightBuffer(sizeof(LightData), capacity, new_buffer)) {
			destroyBuffer(light_buffer.buffer);
			light_buffer.buffer = new_buffer;
			light_buffer.capacity = capacity;
			// Дескриптор в наборе сцены должен указывать на новый буфер
			writeBufferDescriptor(vk_scene_set, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, new_buffer.buffer);
		} else {
			// Новый буфер создать не удалось, рисуем столько источников, сколько помещается в старый
			count = light_buffer.capacity;
		}
	}

	const LightBufferHeader header = { .count = count };
	auto* bytes = static_cast<char*>(light_buffer.buffer.mapped);
	std::memcpy(bytes, &header, sizeof(header));
	if (count > 0) {
		std::memcpy(bytes + sizeof(header), lights.data(), sizeof(LightData) * count);
	}
	vmaFlushAllocation(graphics::internal::context.allocator, light_buffer.buffer.allocation,
	                   0, sizeof(header) + sizeof(LightData) * count);
}

// Записывает вершины визуализации источников в вершинный буфер.
// Если они не помещаются, буфер пересоздаётся в два раза больше, как и storage-буферы.
// Дескриптор здесь обновлять не нужно: вершинный буфер привязывается командой при отрисовке.
// Возвращает false, если рисовать нечего или новый буфер создать не удалось.
bool uploadMarkers() {
	const uint32_t count = uint32_t(marker_vertices.size());
	if (count == 0) {
		return false;
	}

	if (count > marker_buffer.capacity) {
		const uint32_t capacity = std::max(count, marker_buffer.capacity * 2);
		Buffer new_buffer;
		if (!createBuffer(sizeof(MarkerVertex) * capacity, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, new_buffer)) {
			return false;
		}
		destroyBuffer(marker_buffer.buffer);
		marker_buffer.buffer = new_buffer;
		marker_buffer.capacity = capacity;
	}

	writeBuffer(marker_buffer.buffer, marker_vertices.data(), sizeof(MarkerVertex) * count);
	return true;
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

// Два набора дескрипторов (дополнительное задание 1).
// Набор 0 (сцена): uniform-буфер SceneData и два storage-буфера с источниками.
// Набор 1 (модель): uniform-буфер ModelData с матрицей и материалом, у каждого объекта свой.
bool createDescriptors() {
	auto& context = graphics::internal::context;

	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = scene_binding,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = point_lights_binding,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = spot_lights_binding,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(scene_bindings) / sizeof(scene_bindings[0]),
		.pBindings = scene_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr,
	                                &vk_scene_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set layout for scene\n";
		return false;
	}

	const VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr,
	                                &vk_model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set layout for model\n";
		return false;
	}

	// Один набор сцены и по одному набору модели на каждый объект
	const uint32_t object_count = uint32_t(state.objects.size());

	const VkDescriptorPoolSize pool_sizes[] = {
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1 + object_count,
		},
		{
			.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 2,
		},
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + object_count,
		.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr,
	                           &vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor pool\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo scene_set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vk_scene_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &scene_set_info, &vk_scene_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan descriptor set for scene\n";
		return false;
	}

	writeBufferDescriptor(vk_scene_set, scene_binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	                      scene_buffer.buffer);
	writeBufferDescriptor(vk_scene_set, point_lights_binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	                      point_light_buffer.buffer.buffer);
	writeBufferDescriptor(vk_scene_set, spot_lights_binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	                      spot_light_buffer.buffer.buffer);

	// Наборы моделей выделяются все сразу, у всех одна и та же раскладка
	const std::vector<VkDescriptorSetLayout> model_layouts(object_count, vk_model_set_layout);
	std::vector<VkDescriptorSet> model_sets(object_count);

	const VkDescriptorSetAllocateInfo model_set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = object_count,
		.pSetLayouts = model_layouts.data(),
	};

	if (vkAllocateDescriptorSets(context.device, &model_set_info, model_sets.data()) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan descriptor sets for models\n";
		return false;
	}

	for (uint32_t i = 0; i < object_count; ++i) {
		SceneObject& object = state.objects[i];
		object.descriptor_set = model_sets[i];
		writeBufferDescriptor(object.descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		                      object.uniform_buffer.buffer);
	}

	return true;
}

// Параметры, которыми отличаются конвейеры
struct PipelineSettings {
	const char* vertex_shader;
	const char* fragment_shader;
	// Описание вершины, должно совпадать со структурой вершины и входами вершинного шейдера
	uint32_t vertex_stride;
	const VkVertexInputAttributeDescription* attributes;
	uint32_t attribute_count;
	VkPrimitiveTopology topology;
	VkCullModeFlags cull_mode;
};

// Создаёт графический конвейер с общей раскладкой vk_pipeline_layout.
// Возвращает VK_NULL_HANDLE, если создать не получилось.
VkPipeline createGraphicsPipeline(const PipelineSettings& settings) {
	auto& context = graphics::internal::context;

	const VkShaderModule vertex_shader = loadShaderModule(settings.vertex_shader);
	const VkShaderModule fragment_shader = loadShaderModule(settings.fragment_shader);
	if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return VK_NULL_HANDLE;
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

	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = settings.vertex_stride,
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = settings.attribute_count,
		.pVertexAttributeDescriptions = settings.attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = settings.topology,
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
	// Толщина линий 1: для других значений нужна отдельная возможность устройства (wideLines).
	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = settings.cull_mode,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	// Буфер глубины создаётся в стартовом коде и каждый кадр заполняется значением 1.0.
	// Тест глубины включён у всех конвейеров, поэтому объекты закрывают и друг друга, и фигуры источников.
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

	VkPipeline pipeline = VK_NULL_HANDLE;
	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                  &pipeline_info, nullptr, &pipeline);

	// После создания конвейера шейдерные модули больше не нужны
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan graphics pipeline with " << settings.vertex_shader << '\n';
		return VK_NULL_HANDLE;
	}

	return pipeline;
}

bool createPipelines() {
	auto& context = graphics::internal::context;

	// Порядок раскладок задаёт номера наборов: 0 это сцена, 1 это модель
	const VkDescriptorSetLayout set_layouts[] = {
		vk_scene_set_layout,
		vk_model_set_layout,
	};

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(set_layouts) / sizeof(set_layouts[0]),
		.pSetLayouts = set_layouts,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr,
	                           &vk_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		return false;
	}

	// Объекты сцены: у вершины позиция и нормаль, освещение по Блинну-Фонгу
	const VkVertexInputAttributeDescription lit_attributes[] = {
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
	};

	vk_lit_pipeline = createGraphicsPipeline({
		.vertex_shader = "blinn_phong.vert.spv",
		.fragment_shader = "blinn_phong.frag.spv",
		.vertex_stride = sizeof(Vertex),
		.attributes = lit_attributes,
		.attribute_count = sizeof(lit_attributes) / sizeof(lit_attributes[0]),
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.cull_mode = VK_CULL_MODE_BACK_BIT,
	});

	// Визуализация источников (задание 4): у вершины позиция и цвет, шейдеры без освещения
	const VkVertexInputAttributeDescription marker_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(MarkerVertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(MarkerVertex, color),
		},
	};

	// Кубики рисуются треугольниками
	vk_marker_pipeline = createGraphicsPipeline({
		.vertex_shader = "light_marker.vert.spv",
		.fragment_shader = "light_marker.frag.spv",
		.vertex_stride = sizeof(MarkerVertex),
		.attributes = marker_attributes,
		.attribute_count = sizeof(marker_attributes) / sizeof(marker_attributes[0]),
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.cull_mode = VK_CULL_MODE_BACK_BIT,
	});

	// Конусы прожекторов рисуются отрезками, каждые две вершины дают один отрезок.
	// У отрезка нет лицевой и задней стороны, поэтому отсечение выключено.
	vk_marker_line_pipeline = createGraphicsPipeline({
		.vertex_shader = "light_marker.vert.spv",
		.fragment_shader = "light_marker.frag.spv",
		.vertex_stride = sizeof(MarkerVertex),
		.attributes = marker_attributes,
		.attribute_count = sizeof(marker_attributes) / sizeof(marker_attributes[0]),
		.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
		.cull_mode = VK_CULL_MODE_NONE,
	});

	return vk_lit_pipeline != VK_NULL_HANDLE &&
	       vk_marker_pipeline != VK_NULL_HANDLE &&
	       vk_marker_line_pipeline != VK_NULL_HANDLE;
}

void addObject(const char* name, ParallelepipedShape shape, Vec3 position, Material material,
               float spin_speed = 0.0f) {
	SceneObject object;
	object.name = name;
	object.shape = shape;
	object.position = position;
	object.material = material;
	object.spin_speed = spin_speed;
	state.objects.push_back(object);
}

// Начальная сцена: параллелепипед по варианту, пол, ещё два параллелепипеда и источники света
void createScene() {
	addObject("Параллелепипед (вариант 3)", { .a = 2.0f, .b = 1.2f, .c = 0.8f }, { 0.0f, 0.6f, 0.0f },
	          { .albedo = { 0.9f, 0.42f, 0.18f }, .specular_color = { 0.8f, 0.8f, 0.8f }, .shininess = 64.0f },
	          20.0f);
	addObject("Пол", { .a = 20.0f, .b = 0.2f, .c = 20.0f }, { 0.0f, -0.1f, 0.0f },
	          { .albedo = { 0.5f, 0.5f, 0.52f }, .specular_color = { 0.15f, 0.15f, 0.15f }, .shininess = 16.0f });
	addObject("Наклонный параллелепипед",
	          { .a = 1.2f, .b = 1.6f, .c = 1.0f, .alpha = 70.0f, .beta = 110.0f, .gamma = 60.0f },
	          { -3.2f, 0.99f, -1.2f },
	          { .albedo = { 0.2f, 0.45f, 0.85f }, .specular_color = { 1.0f, 1.0f, 1.0f }, .shininess = 128.0f });
	addObject("Колонна", { .a = 0.7f, .b = 3.0f, .c = 0.7f }, { 3.0f, 1.5f, -1.5f },
	          { .albedo = { 0.85f, 0.82f, 0.75f }, .specular_color = { 0.4f, 0.4f, 0.4f }, .shininess = 24.0f });

	state.point_lights.push_back({ .position = { -2.2f, 1.3f, 1.8f }, .color = { 1.0f, 0.5f, 0.2f }, .intensity = 4.0f });
	state.point_lights.push_back({ .position = { 2.0f, 0.9f, 1.8f }, .color = { 0.25f, 0.55f, 1.0f }, .intensity = 4.0f });

	state.spot_lights.push_back({
		.position = { 0.0f, 4.5f, 2.0f }, .yaw = 0.0f, .pitch = -63.0f,
		.inner_angle = 12.0f, .outer_angle = 20.0f, .color = { 1.0f, 0.95f, 0.8f }, .intensity = 14.0f,
	});
	state.spot_lights.push_back({
		.position = { 5.0f, 4.0f, 1.5f }, .yaw = -34.0f, .pitch = -35.0f,
		.inner_angle = 8.0f, .outer_angle = 16.0f, .color = { 0.6f, 1.0f, 0.4f }, .intensity = 12.0f,
	});
}

bool createResources() {
	createScene();

	for (SceneObject& object : state.objects) {
		if (!computeEdges(object.shape, object.edges)) {
			std::cerr << "Default shape of " << object.name << " is invalid\n";
			return false;
		}

		if (!createBuffer(sizeof(Vertex) * vertex_count, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, object.vertex_buffer) ||
		    !createBuffer(sizeof(uint16_t) * index_count, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, object.index_buffer) ||
		    !createBuffer(sizeof(ModelData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, object.uniform_buffer)) {
			return false;
		}

		uploadGeometry(object);
	}

	if (!createBuffer(sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_buffer) ||
	    !createLightBuffer(sizeof(PointLightData), initial_light_capacity, point_light_buffer.buffer) ||
	    !createLightBuffer(sizeof(SpotLightData), initial_light_capacity, spot_light_buffer.buffer)) {
		return false;
	}
	point_light_buffer.capacity = initial_light_capacity;
	spot_light_buffer.capacity = initial_light_capacity;

	// Кубик для визуализации источников строится тем же генератором, что и параллелепипеды.
	// Индексы сразу разворачиваются в список из 36 вершин треугольников.
	const Vec3 cube_edges[3] = {
		{ marker_cube_size, 0.0f, 0.0f },
		{ 0.0f, marker_cube_size, 0.0f },
		{ 0.0f, 0.0f, marker_cube_size },
	};
	Vertex cube_vertices[vertex_count];
	uint16_t cube_indices[index_count];
	generateParallelepiped(cube_edges, cube_vertices, cube_indices);
	for (uint32_t i = 0; i < index_count; ++i) {
		marker_cube[i] = cube_vertices[cube_indices[i]].position;
	}

	return createDescriptors() && createPipelines();
}

// Управление камерой мышью и клавиатурой. Не срабатывает, когда мышь или клавиатура заняты интерфейсом.
void updateCameraInput(float dt) {
	const ImGuiIO& io = ImGui::GetIO();
	Camera& camera = state.camera;

	// Поворот: зажать правую кнопку мыши над сценой и двигать мышь
	if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !io.WantCaptureMouse) {
		state.camera_rotating = true;
	}
	if (!ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
		state.camera_rotating = false;
	}
	if (state.camera_rotating) {
		const float sensitivity = 0.15f; // градусов на пиксель
		camera.yaw += io.MouseDelta.x * sensitivity;
		camera.pitch -= io.MouseDelta.y * sensitivity;
	}

	// Колесо мыши меняет угол обзора
	if (!io.WantCaptureMouse) {
		camera.fov -= io.MouseWheel * 2.0f;
	}

	// Движение: W и S вперёд и назад, A и D влево и вправо, E и Q вверх и вниз, Shift ускоряет
	if (!io.WantCaptureKeyboard) {
		const Vec3 forward = directionFromAngles(camera.yaw, camera.pitch);
		const Vec3 right = normalize(cross(forward, world_up));

		// 1, если клавиша зажата, иначе 0
		auto key = [](ImGuiKey k) { return ImGui::IsKeyDown(k) ? 1.0f : 0.0f; };
		const Vec3 move = forward * (key(ImGuiKey_W) - key(ImGuiKey_S)) +
		                  right * (key(ImGuiKey_D) - key(ImGuiKey_A)) +
		                  world_up * (key(ImGuiKey_E) - key(ImGuiKey_Q));

		if (length(move) > 0.0f) {
			const float speed = camera.speed * (ImGui::IsKeyDown(ImGuiKey_LeftShift) ? 3.0f : 1.0f);
			camera.position = camera.position + normalize(move) * (speed * dt);
		}
	}

	// Углы держим в пределах ползунков. Наклон не доходит до 90 градусов,
	// иначе направление взгляда совпадёт с вертикалью и lookAt не сможет построить оси камеры.
	if (camera.yaw > 180.0f) {
		camera.yaw -= 360.0f;
	} else if (camera.yaw < -180.0f) {
		camera.yaw += 360.0f;
	}
	camera.pitch = std::clamp(camera.pitch, -89.0f, 89.0f);
	camera.fov = std::clamp(camera.fov, 20.0f, 120.0f);
}

// Считает данные сцены, массивы источников и данные каждого объекта
void updateFrameData() {
	const VkExtent2D extent = graphics::internal::context.swapchain_extent;
	const float aspect = extent.height > 0 ? float(extent.width) / float(extent.height) : 1.0f;

	// Матрица вида: камера в своей позиции смотрит вдоль направления, заданного углами
	const Camera& camera = state.camera;
	const Vec3 forward = directionFromAngles(camera.yaw, camera.pitch);
	const Mat4 view = lookAt(camera.position, camera.position + forward, world_up);
	const Mat4 projection = perspective(radians(camera.fov), aspect, camera.near_plane, camera.far_plane);

	scene_data.view_projection = projection * view;
	scene_data.camera_position = camera.position;
	scene_data.ambient_color = state.ambient.color;
	scene_data.ambient_intensity = state.ambient.intensity;
	scene_data.light_direction = directionFromAngles(state.directional.yaw, state.directional.pitch);
	scene_data.light_intensity = state.directional.enabled ? state.directional.intensity : 0.0f;
	scene_data.light_color = state.directional.color;

	// В storage-буферы попадают только включённые источники
	point_light_data.clear();
	for (const PointLight& light : state.point_lights) {
		if (light.enabled) {
			point_light_data.push_back({
				.position = light.position,
				.intensity = light.intensity,
				.color = light.color,
			});
		}
	}

	spot_light_data.clear();
	for (const SpotLight& light : state.spot_lights) {
		if (light.enabled) {
			// Фонарик повторяет позицию и направление камеры
			spot_light_data.push_back({
				.position = light.attached_to_camera ? camera.position : light.position,
				.intensity = light.intensity,
				.direction = light.attached_to_camera ? forward : directionFromAngles(light.yaw, light.pitch),
				.cos_inner = std::cos(radians(light.inner_angle)),
				.color = light.color,
				.cos_outer = std::cos(radians(light.outer_angle)),
			});
		}
	}

	for (SceneObject& object : state.objects) {
		// Матрица модели: сначала растяжение, потом вращение вокруг вертикали,
		// потом поворот из интерфейса и перенос
		const Vec3 rotation_radians = {
			radians(object.rotation.x), radians(object.rotation.y), radians(object.rotation.z),
		};
		const Mat4 rotation = rotateEuler(rotation_radians) * rotateY(radians(object.spin_angle));
		const Mat4 model = translate(object.position) * rotation * scale(object.scale);

		// Матрица для нормалей: поворот, умноженный на обратное растяжение.
		// Так нормали остаются перпендикулярными граням даже при неравномерном растяжении.
		const Mat4 normal_matrix = rotation * scale({
			1.0f / object.scale.x, 1.0f / object.scale.y, 1.0f / object.scale.z,
		});

		object.model_data.transform = model;
		object.model_data.normal_matrix = normal_matrix;
		object.model_data.albedo = object.material.albedo;
		object.model_data.shininess = object.material.shininess;
		object.model_data.specular_color = object.material.specular_color;
	}
}

// Строит вершины визуализации источников (дополнительное задание 4).
// Фигуры строятся на процессоре сразу в мировых координатах и потом не освещаются.
// Выключенные источники и фонарик (он находится в камере) не рисуются.
void buildLightMarkers() {
	marker_vertices.clear();

	// Кубик цвета источника вокруг позиции каждого точечного источника и прожектора
	if (state.show_light_cubes) {
		auto addCube = [](Vec3 center, Vec3 color) {
			for (const Vec3& corner : marker_cube) {
				marker_vertices.push_back({ center + corner, color });
			}
		};
		for (const PointLight& light : state.point_lights) {
			if (light.enabled) {
				addCube(light.position, light.color);
			}
		}
		for (const SpotLight& light : state.spot_lights) {
			if (light.enabled && !light.attached_to_camera) {
				addCube(light.position, light.color);
			}
		}
	}
	marker_triangle_vertex_count = uint32_t(marker_vertices.size());

	// Конус прожектора отрезками: белые ось, образующие и окружность внешнего угла,
	// серая окружность внутреннего угла
	if (state.show_spot_cones) {
		const Vec3 white = { 1.0f, 1.0f, 1.0f };
		const Vec3 gray = { 0.45f, 0.45f, 0.45f };
		auto addLine = [](Vec3 a, Vec3 b, Vec3 color) {
			marker_vertices.push_back({ a, color });
			marker_vertices.push_back({ b, color });
		};

		for (const SpotLight& light : state.spot_lights) {
			if (!light.enabled || light.attached_to_camera) {
				continue;
			}

			const Vec3 apex = light.position;
			const Vec3 axis = directionFromAngles(light.yaw, light.pitch);
			// Два единичных вектора, перпендикулярных оси и друг другу.
			// Вспомогательный вектор не должен быть параллелен оси, иначе векторное произведение будет нулевым.
			const Vec3 helper = std::fabs(axis.y) < 0.99f ? Vec3{ 0.0f, 1.0f, 0.0f } : Vec3{ 1.0f, 0.0f, 0.0f };
			const Vec3 u = normalize(cross(axis, helper));
			const Vec3 v = cross(axis, u);
			const float length = state.cone_length;

			// Точка конуса под углом angle к оси, повёрнутая вокруг оси на угол t.
			// Все образующие одной длины, поэтому даже широкий конус остаётся конечным.
			auto conePoint = [&](float angle, float t) {
				const Vec3 side = u * std::cos(t) + v * std::sin(t);
				return apex + (axis * std::cos(angle) + side * std::sin(angle)) * length;
			};

			const float outer = radians(light.outer_angle);
			const float inner = radians(light.inner_angle);

			addLine(apex, apex + axis * length, white);
			for (int k = 0; k < cone_side_lines; ++k) {
				const float t = 2.0f * pi * float(k) / float(cone_side_lines);
				addLine(apex, conePoint(outer, t), white);
			}
			for (int k = 0; k < cone_circle_segments; ++k) {
				const float t0 = 2.0f * pi * float(k) / float(cone_circle_segments);
				const float t1 = 2.0f * pi * float(k + 1) / float(cone_circle_segments);
				addLine(conePoint(outer, t0), conePoint(outer, t1), white);
				addLine(conePoint(inner, t0), conePoint(inner, t1), gray);
			}
		}
	}
}

void drawCameraInterface() {
	Camera& camera = state.camera;
	ImGui::PushID("camera");

	// AlwaysClamp не даёт ввести через Ctrl + клик значение за пределами ползунка.
	// Например, при наклоне 90 градусов lookAt не построит оси камеры.
	const ImGuiSliderFlags clamp = ImGuiSliderFlags_AlwaysClamp;
	ImGui::DragFloat3("Позиция", &camera.position.x, 0.05f, -50.0f, 50.0f, "%.2f");
	ImGui::SliderFloat("Азимут", &camera.yaw, -180.0f, 180.0f, "%.0f°", clamp);
	ImGui::SliderFloat("Наклон", &camera.pitch, -89.0f, 89.0f, "%.0f°", clamp);
	ImGui::SliderFloat("Угол обзора", &camera.fov, 20.0f, 120.0f, "%.0f°", clamp);
	ImGui::SliderFloat("Ближняя плоскость", &camera.near_plane, 0.01f, 5.0f, "%.2f", clamp);
	ImGui::SliderFloat("Дальняя плоскость", &camera.far_plane, 10.0f, 200.0f, "%.0f", clamp);
	ImGui::SliderFloat("Скорость движения", &camera.speed, 0.5f, 20.0f, "%.1f", clamp);
	if (ImGui::Button("Сбросить камеру")) {
		camera = Camera{};
	}
	ImGui::TextDisabled("Правая кнопка мыши + движение мыши: поворот");
	ImGui::TextDisabled("W, A, S, D: движение, E и Q: вверх и вниз, Shift: быстрее");
	ImGui::TextDisabled("Колесо мыши: угол обзора");

	ImGui::PopID();
}

void drawAmbientInterface() {
	const ImGuiSliderFlags clamp = ImGuiSliderFlags_AlwaysClamp;
	ImGui::PushID("ambient");
	ImGui::ColorEdit3("Цвет", &state.ambient.color.x);
	ImGui::SliderFloat("Интенсивность", &state.ambient.intensity, 0.0f, 1.0f, "%.2f", clamp);
	ImGui::PopID();
}

void drawDirectionalInterface() {
	const ImGuiSliderFlags clamp = ImGuiSliderFlags_AlwaysClamp;
	DirectionalLight& light = state.directional;
	ImGui::PushID("directional");
	ImGui::Checkbox("Включён", &light.enabled);
	ImGui::ColorEdit3("Цвет", &light.color.x);
	ImGui::SliderFloat("Интенсивность", &light.intensity, 0.0f, 5.0f, "%.2f", clamp);
	ImGui::SliderFloat("Азимут", &light.yaw, -180.0f, 180.0f, "%.0f°", clamp);
	ImGui::SliderFloat("Наклон", &light.pitch, -90.0f, 90.0f, "%.0f°", clamp);
	ImGui::TextDisabled("Углы задают направление, в котором светит источник");
	ImGui::PopID();
}

void drawObjectsInterface() {
	ImGui::PushID("objects");

	if (ImGui::BeginCombo("Объект", state.objects[state.selected_object].name)) {
		for (int i = 0; i < int(state.objects.size()); ++i) {
			const bool selected = i == state.selected_object;
			if (ImGui::Selectable(state.objects[i].name, selected)) {
				state.selected_object = i;
			}
			if (selected) {
				ImGui::SetItemDefaultFocus();
			}
		}
		ImGui::EndCombo();
	}

	SceneObject& object = state.objects[state.selected_object];
	ImGui::PushID(state.selected_object);

	// Угол γ = 0 или 180 градусов дал бы деление на ноль в computeEdges, поэтому углы ограничены
	const ImGuiSliderFlags clamp = ImGuiSliderFlags_AlwaysClamp;

	ImGui::SeparatorText("Форма");
	ParallelepipedShape& shape = object.shape;
	bool changed = false;
	changed |= ImGui::DragFloat("Ребро a", &shape.a, 0.01f, 0.1f, 30.0f, "%.2f", clamp);
	changed |= ImGui::DragFloat("Ребро b", &shape.b, 0.01f, 0.1f, 30.0f, "%.2f", clamp);
	changed |= ImGui::DragFloat("Ребро c", &shape.c, 0.01f, 0.1f, 30.0f, "%.2f", clamp);
	changed |= ImGui::SliderFloat("Угол α (b, c)", &shape.alpha, 30.0f, 150.0f, "%.0f°", clamp);
	changed |= ImGui::SliderFloat("Угол β (a, c)", &shape.beta, 30.0f, 150.0f, "%.0f°", clamp);
	changed |= ImGui::SliderFloat("Угол γ (a, b)", &shape.gamma, 30.0f, 150.0f, "%.0f°", clamp);
	if (ImGui::Button("Прямоугольный")) {
		shape.alpha = shape.beta = shape.gamma = 90.0f;
		changed = true;
	}

	// Если форма изменилась, пересчитываем рёбра. Вершинный буфер обновится в render().
	if (changed) {
		Vec3 edges[3];
		object.shape_valid = computeEdges(shape, edges);
		if (object.shape_valid) {
			std::memcpy(object.edges, edges, sizeof(edges));
			object.geometry_dirty = true;
		}
	}

	if (!object.shape_valid) {
		ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Такие углы не образуют параллелепипед");
	}

	ImGui::SeparatorText("Положение");
	ImGui::DragFloat3("Позиция", &object.position.x, 0.02f, -20.0f, 20.0f, "%.2f");
	ImGui::SliderFloat3("Поворот X/Y/Z", &object.rotation.x, -180.0f, 180.0f, "%.0f°");
	// Растяжение не меньше 0.05 даже при вводе с клавиатуры.
	// При нуле или отрицательном значении ломается матрица нормалей и переворачиваются грани.
	ImGui::DragFloat3("Растяжение", &object.scale.x, 0.01f, 0.05f, 5.0f, "%.2f", clamp);
	ImGui::SliderFloat("Вращение", &object.spin_speed, -180.0f, 180.0f, "%.0f°/с");

	// Коэффициент блика не меньше 1: pow(0, 0) в шейдере не определён
	ImGui::SeparatorText("Материал");
	ImGui::ColorEdit3("Альбедо", &object.material.albedo.x);
	ImGui::ColorEdit3("Цвет блика", &object.material.specular_color.x);
	ImGui::SliderFloat("Коэффициент блика", &object.material.shininess, 1.0f, 512.0f, "%.0f",
	                   clamp | ImGuiSliderFlags_Logarithmic);

	ImGui::PopID();
	ImGui::PopID();
}

void drawPointLightsInterface() {
	ImGui::PushID("point");

	if (ImGui::Button("Добавить точечный источник")) {
		state.point_lights.push_back(PointLight{});
	}
	ImGui::TextDisabled("Включено %d, буфер вмещает %u (растёт сам)",
	                    int(point_light_data.size()), point_light_buffer.capacity);

	int remove_index = -1;
	for (int i = 0; i < int(state.point_lights.size()); ++i) {
		PointLight& light = state.point_lights[i];
		ImGui::PushID(i);
		if (ImGui::TreeNodeEx("light", ImGuiTreeNodeFlags_DefaultOpen, "Точечный источник %d", i + 1)) {
			ImGui::Checkbox("Включён", &light.enabled);
			ImGui::DragFloat3("Позиция", &light.position.x, 0.05f, -50.0f, 50.0f, "%.2f");
			ImGui::ColorEdit3("Цвет", &light.color.x);
			ImGui::SliderFloat("Интенсивность", &light.intensity, 0.0f, 100.0f, "%.1f",
			                   ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_Logarithmic);
			if (ImGui::Button("Удалить")) {
				remove_index = i;
			}
			ImGui::TreePop();
		}
		ImGui::PopID();
	}

	// Удаляем после цикла, чтобы не менять массив во время обхода
	if (remove_index >= 0) {
		state.point_lights.erase(state.point_lights.begin() + remove_index);
	}

	ImGui::PopID();
}

void drawSpotLightsInterface() {
	ImGui::PushID("spot");

	if (ImGui::Button("Добавить прожектор")) {
		state.spot_lights.push_back(SpotLight{});
	}
	ImGui::TextDisabled("Включено %d, буфер вмещает %u (растёт сам)",
	                    int(spot_light_data.size()), spot_light_buffer.capacity);

	int remove_index = -1;
	for (int i = 0; i < int(state.spot_lights.size()); ++i) {
		SpotLight& light = state.spot_lights[i];
		ImGui::PushID(i);
		if (ImGui::TreeNodeEx("light", ImGuiTreeNodeFlags_DefaultOpen, "Прожектор %d", i + 1)) {
			ImGui::Checkbox("Включён", &light.enabled);
			ImGui::Checkbox("Прикрепить к камере (фонарик)", &light.attached_to_camera);

			const ImGuiSliderFlags clamp = ImGuiSliderFlags_AlwaysClamp;

			// У фонарика позицию и направление задаёт камера
			ImGui::BeginDisabled(light.attached_to_camera);
			ImGui::DragFloat3("Позиция", &light.position.x, 0.05f, -50.0f, 50.0f, "%.2f");
			ImGui::SliderFloat("Азимут", &light.yaw, -180.0f, 180.0f, "%.0f°", clamp);
			ImGui::SliderFloat("Наклон", &light.pitch, -90.0f, 90.0f, "%.0f°", clamp);
			ImGui::EndDisabled();

			ImGui::ColorEdit3("Цвет", &light.color.x);
			ImGui::SliderFloat("Интенсивность", &light.intensity, 0.0f, 100.0f, "%.1f",
			                   clamp | ImGuiSliderFlags_Logarithmic);

			// Углы отсчитываются от оси конуса. Внешний угол не может быть меньше внутреннего.
			if (ImGui::SliderFloat("Внутренний угол", &light.inner_angle, 0.0f, 89.0f, "%.1f°", clamp)) {
				light.outer_angle = std::max(light.outer_angle, light.inner_angle);
			}
			if (ImGui::SliderFloat("Внешний угол", &light.outer_angle, 0.0f, 89.0f, "%.1f°", clamp)) {
				light.inner_angle = std::min(light.inner_angle, light.outer_angle);
			}

			if (ImGui::Button("Удалить")) {
				remove_index = i;
			}
			ImGui::TreePop();
		}
		ImGui::PopID();
	}

	if (remove_index >= 0) {
		state.spot_lights.erase(state.spot_lights.begin() + remove_index);
	}

	ImGui::PopID();
}

void drawMarkersInterface() {
	ImGui::PushID("markers");
	ImGui::Checkbox("Показывать кубики источников", &state.show_light_cubes);
	ImGui::Checkbox("Показывать конусы прожекторов", &state.show_spot_cones);
	ImGui::SliderFloat("Длина конуса", &state.cone_length, 0.2f, 5.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
	ImGui::TextDisabled("Кубик имеет цвет источника и не освещается");
	ImGui::TextDisabled("Белые линии: ось и внешний угол, серая окружность: внутренний угол");
	ImGui::PopID();
}

// Панель ImGui со всеми настройками
void drawInterface() {
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(430.0f, 700.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Лабораторная работа №2");

	ImGui::Text("Вариант 3: параллелепипед. FPS: %.0f", ImGui::GetIO().Framerate);

	if (ImGui::CollapsingHeader("Камера", ImGuiTreeNodeFlags_DefaultOpen)) {
		drawCameraInterface();
	}
	if (ImGui::CollapsingHeader("Фоновый свет", ImGuiTreeNodeFlags_DefaultOpen)) {
		drawAmbientInterface();
	}
	if (ImGui::CollapsingHeader("Направленный источник", ImGuiTreeNodeFlags_DefaultOpen)) {
		drawDirectionalInterface();
	}
	if (ImGui::CollapsingHeader("Объекты и материалы", ImGuiTreeNodeFlags_DefaultOpen)) {
		drawObjectsInterface();
	}
	// Дополнительное задание 2
	if (ImGui::CollapsingHeader("2. Точечные источники", ImGuiTreeNodeFlags_DefaultOpen)) {
		drawPointLightsInterface();
	}
	// Дополнительное задание 3
	if (ImGui::CollapsingHeader("3. Прожекторы", ImGuiTreeNodeFlags_DefaultOpen)) {
		drawSpotLightsInterface();
	}
	// Дополнительное задание 4
	if (ImGui::CollapsingHeader("4. Визуализация источников", ImGuiTreeNodeFlags_DefaultOpen)) {
		drawMarkersInterface();
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

	vkDestroyPipeline(context.device, vk_marker_line_pipeline, nullptr);
	vkDestroyPipeline(context.device, vk_marker_pipeline, nullptr);
	vkDestroyPipeline(context.device, vk_lit_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr); // наборы дескрипторов удаляются вместе с пулом
	vkDestroyDescriptorSetLayout(context.device, vk_model_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_scene_set_layout, nullptr);

	destroyBuffer(marker_buffer.buffer);
	destroyBuffer(spot_light_buffer.buffer);
	destroyBuffer(point_light_buffer.buffer);
	destroyBuffer(scene_buffer);

	for (SceneObject& object : state.objects) {
		destroyBuffer(object.uniform_buffer);
		destroyBuffer(object.index_buffer);
		destroyBuffer(object.vertex_buffer);
	}
}

void update(double time) {
	if (state.previous_time < 0.0) {
		state.previous_time = time;
	}
	float dt = float(time - state.previous_time);
	state.previous_time = time;
	if (dt > 0.1f) {
		dt = 0.1f; // чтобы камера и объекты не прыгали после перетаскивания окна или остановки в отладчике
	}

	updateCameraInput(dt);

	for (SceneObject& object : state.objects) {
		object.spin_angle = std::fmod(object.spin_angle + object.spin_speed * dt, 360.0f);
	}

	drawInterface();
	updateFrameData();
	buildLightMarkers();
}

void render(const graphics::internal::FrameData& fd) {
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	auto& context = graphics::internal::context;

	// prepare() уже дождался конца прошлого кадра, значит GPU больше не читает эти буферы
	// и их можно перезаписать или пересоздать. В update() этого делать нельзя.
	for (SceneObject& object : state.objects) {
		if (object.geometry_dirty) {
			uploadGeometry(object);
			object.geometry_dirty = false;
		}
		writeBuffer(object.uniform_buffer, &object.model_data, sizeof(ModelData));
	}
	writeBuffer(scene_buffer, &scene_data, sizeof(scene_data));
	uploadLights(point_light_buffer, point_lights_binding, point_light_data);
	uploadLights(spot_light_buffer, spot_lights_binding, spot_light_data);
	const bool draw_markers = uploadMarkers();

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

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_lit_pipeline);

	// Набор сцены привязывается один раз на кадр
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
	                        0, 1, &vk_scene_set, 0, nullptr);

	// Для каждого объекта меняется только набор модели и буферы геометрии
	for (const SceneObject& object : state.objects) {
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
		                        1, 1, &object.descriptor_set, 0, nullptr);

		const VkDeviceSize vertex_offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &object.vertex_buffer.buffer, &vertex_offset);
		vkCmdBindIndexBuffer(cmd, object.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);

		vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
	}

	// Визуализация источников (задание 4). У конвейеров та же раскладка,
	// поэтому набор сцены с матрицей view-projection остаётся привязанным.
	if (draw_markers) {
		const VkDeviceSize marker_offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &marker_buffer.buffer.buffer, &marker_offset);

		// Сначала в буфере лежат треугольники кубиков, за ними отрезки конусов
		if (marker_triangle_vertex_count > 0) {
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_marker_pipeline);
			vkCmdDraw(cmd, marker_triangle_vertex_count, 1, 0, 0);
		}

		const uint32_t line_vertex_count = uint32_t(marker_vertices.size()) - marker_triangle_vertex_count;
		if (line_vertex_count > 0) {
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_marker_line_pipeline);
			vkCmdDraw(cmd, line_vertex_count, 1, marker_triangle_vertex_count, 0);
		}
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
