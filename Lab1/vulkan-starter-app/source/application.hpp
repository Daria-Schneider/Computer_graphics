#pragma once

#include <cstddef>
#include <cstdint>

#include "graphics_internal.hpp"
#include "math3d.hpp"

namespace application {

// Вершина параллелепипеда: позиция, нормаль и цвет, по 3 float.
// Размер 36 байт, правила std140 к вершинам не относятся.
struct Vertex {
	math::Vec3 position; // location = 0
	math::Vec3 normal; // location = 1
	math::Vec3 color; // location = 2
};

// Копия uniform-блока из шейдеров. Раскладка std140:
// mat4 занимает 64 байта, vec4 16 байт, uint 4 байта,
// vec4 и mat4 должны начинаться с адреса, кратного 16.
struct alignas(16) UniformData {
	math::Mat4 mvp; // смещение 0, произведение проекции, вида и модели
	math::Mat4 normal_matrix; // смещение 64, матрица для нормалей
	math::Vec4 color; // смещение 128, цвет из интерфейса
	uint32_t use_vertex_colors; // смещение 144, 1 если умножать на цвета вершин
	uint32_t use_lighting; // смещение 148, 1 если включено затенение
	uint32_t padding[2]; // добивка до 160 байт, размер блока должен быть кратен 16
};

// Проверка, что смещения совпадают с шейдером
static_assert(sizeof(Vertex) == 36);
static_assert(offsetof(UniformData, normal_matrix) == 64);
static_assert(offsetof(UniformData, color) == 128);
static_assert(offsetof(UniformData, use_vertex_colors) == 144);
static_assert(offsetof(UniformData, use_lighting) == 148);
static_assert(sizeof(UniformData) == 160);

bool initialize();
void shutdown();

void update(double time);
void render(const graphics::internal::FrameData& fd);

} // namespace application
