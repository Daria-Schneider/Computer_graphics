#version 450

// Вершина фигуры визуализации источника, порядок как в struct MarkerVertex
layout(location = 0) in vec3 in_position; // позиция сразу в мировых координатах
layout(location = 1) in vec3 in_color;

// Тот же блок данных сцены, что и в blinn_phong.vert. Нужна только матрица view_projection.
layout(set = 0, binding = 0, std140) uniform SceneData {
	mat4 view_projection;
	vec3 camera_position;
	float padding0;
	vec3 ambient_color;
	float ambient_intensity;
	vec3 light_direction;
	float light_intensity;
	vec3 light_color;
	float padding1;
} scene;

layout(location = 0) out vec3 out_color;

void main() {
	gl_Position = scene.view_projection * vec4(in_position, 1.0);
	out_color = in_color;
}
