#pragma once

#include <cstddef>
#include <cstdint>

#include "graphics_internal.hpp"
#include "math3d.hpp"

namespace application {

// Вершина параллелепипеда: позиция и нормаль, по 3 float.
// Размер 24 байта, правила std140 и std430 к вершинам не относятся.
struct Vertex {
	math::Vec3 position; // location = 0
	math::Vec3 normal; // location = 1
};

// Вершина фигур визуализации источников (дополнительное задание 4):
// позиция сразу в мировых координатах и цвет, по 3 float. Размер 24 байта.
struct MarkerVertex {
	math::Vec3 position; // location = 0
	math::Vec3 color; // location = 1
};

// Данные сцены, общие для всех объектов. Набор дескрипторов 0, binding 0, раскладка std140.
// В std140 vec3 начинается с адреса, кратного 16, и занимает 12 байт,
// поэтому после каждого vec3 стоит float: это или нужный параметр, или добивка.
struct alignas(16) SceneData {
	math::Mat4 view_projection; // смещение 0, произведение матриц проекции и вида
	math::Vec3 camera_position; // смещение 64, позиция камеры в мировых координатах
	float padding0; // смещение 76, добивка
	math::Vec3 ambient_color; // смещение 80, цвет фонового света
	float ambient_intensity; // смещение 92, интенсивность фонового света
	math::Vec3 light_direction; // смещение 96, куда светит направленный источник
	float light_intensity; // смещение 108, интенсивность направленного источника
	math::Vec3 light_color; // смещение 112, цвет направленного источника
	float padding1; // смещение 124, добивка до 128 байт
};

// Данные одного объекта. Набор дескрипторов 1, binding 0, раскладка std140.
struct alignas(16) ModelData {
	math::Mat4 transform; // смещение 0, матрица модели
	math::Mat4 normal_matrix; // смещение 64, матрица для нормалей
	math::Vec3 albedo; // смещение 128, цвет рассеянного отражения
	float shininess; // смещение 140, коэффициент блика
	math::Vec3 specular_color; // смещение 144, цвет блика
	float padding; // смещение 156, добивка до 160 байт
};

// Точечный источник в storage-буфере, раскладка std430. Размер 32 байта.
struct alignas(16) PointLightData {
	math::Vec3 position; // смещение 0
	float intensity; // смещение 12
	math::Vec3 color; // смещение 16
	float padding; // смещение 28
};

// Прожектор в storage-буфере, раскладка std430. Размер 48 байт.
struct alignas(16) SpotLightData {
	math::Vec3 position; // смещение 0
	float intensity; // смещение 12
	math::Vec3 direction; // смещение 16, единичный вектор вдоль оси конуса
	float cos_inner; // смещение 28, косинус внутреннего угла
	math::Vec3 color; // смещение 32
	float cos_outer; // смещение 44, косинус внешнего угла
};

// Начало storage-буфера с источниками: их количество.
// Массив источников идёт со смещения 16, потому что структуры источников выравниваются по 16 байт.
struct alignas(16) LightBufferHeader {
	uint32_t count;
	uint32_t padding[3];
};

// Проверка, что смещения совпадают с шейдерами
static_assert(sizeof(Vertex) == 24);
static_assert(sizeof(MarkerVertex) == 24);

static_assert(offsetof(SceneData, camera_position) == 64);
static_assert(offsetof(SceneData, ambient_color) == 80);
static_assert(offsetof(SceneData, ambient_intensity) == 92);
static_assert(offsetof(SceneData, light_direction) == 96);
static_assert(offsetof(SceneData, light_intensity) == 108);
static_assert(offsetof(SceneData, light_color) == 112);
static_assert(sizeof(SceneData) == 128);

static_assert(offsetof(ModelData, normal_matrix) == 64);
static_assert(offsetof(ModelData, albedo) == 128);
static_assert(offsetof(ModelData, shininess) == 140);
static_assert(offsetof(ModelData, specular_color) == 144);
static_assert(sizeof(ModelData) == 160);

static_assert(offsetof(PointLightData, intensity) == 12);
static_assert(offsetof(PointLightData, color) == 16);
static_assert(sizeof(PointLightData) == 32);

static_assert(offsetof(SpotLightData, intensity) == 12);
static_assert(offsetof(SpotLightData, direction) == 16);
static_assert(offsetof(SpotLightData, cos_inner) == 28);
static_assert(offsetof(SpotLightData, color) == 32);
static_assert(offsetof(SpotLightData, cos_outer) == 44);
static_assert(sizeof(SpotLightData) == 48);

static_assert(sizeof(LightBufferHeader) == 16);

bool initialize();
void shutdown();

void update(double time);
void render(const graphics::internal::FrameData& fd);

} // namespace application
