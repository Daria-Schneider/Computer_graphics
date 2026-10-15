#version 450

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_normal;

// Такой же блок, как в вершинном шейдере
layout(set = 0, binding = 0, std140) uniform UniformData {
	mat4 mvp;
	mat4 normal_matrix;
	vec4 color;
	uint use_vertex_colors;
	uint use_lighting;
} ubo;

layout(location = 0) out vec4 out_color;

void main() {
	// Итоговый цвет = цвет вершины * цвет из интерфейса
	vec3 vertex_color = ubo.use_vertex_colors != 0u ? in_color : vec3(1.0);
	vec3 color = vertex_color * ubo.color.rgb;

	// Простое затенение, чтобы грани одного цвета не сливались
	if (ubo.use_lighting != 0u) {
		const vec3 light_direction = normalize(vec3(0.4, 0.8, 0.6));
		float diffuse = max(dot(normalize(in_normal), light_direction), 0.0);
		color *= 0.35 + 0.65 * diffuse;
	}

	out_color = vec4(color, ubo.color.a);
}
