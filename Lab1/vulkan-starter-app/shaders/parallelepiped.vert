#version 450

// Входные данные вершины, порядок как в struct Vertex
layout(location = 0) in vec3 in_position; // позиция в локальных координатах
layout(location = 1) in vec3 in_normal; // нормаль грани
layout(location = 2) in vec3 in_color; // цвет вершины

// Uniform-буфер, должен совпадать со struct UniformData в application.hpp
layout(set = 0, binding = 0, std140) uniform UniformData {
	mat4 mvp;
	mat4 normal_matrix; // mat4 вместо mat3, потому что mat3 в std140 выравнивается неудобно
	vec4 color; // цвет из интерфейса
	uint use_vertex_colors;
	uint use_lighting;
} ubo;

layout(location = 0) out vec3 out_color;
layout(location = 1) out vec3 out_normal;

void main() {
	gl_Position = ubo.mvp * vec4(in_position, 1.0);
	out_color = in_color;
	out_normal = mat3(ubo.normal_matrix) * in_normal;
}
