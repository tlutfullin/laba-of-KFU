#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 10 // Размерность матриц

// Функция для генерации случайной матрицы размером SIZE x SIZE
void generate_matrix(int matrix[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix[i][j] = rand() % 100; // случайное значение от 0 до 99
        }
    }
}

// Функция для умножения двух матриц
void multiply_matrices(int result[SIZE][SIZE], int matrix1[SIZE][SIZE], int matrix2[SIZE][SIZE]) {
    #pragma omp parallel for
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result[i][j] = 0;
            for (int k = 0; k < SIZE; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

// Функция для вывода матрицы на экран
void print_matrix(int matrix[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    // Инициализация генератора случайных чисел
    srand(1234);

    // Создание и заполнение случайными значениями матрицы A и B
    int matrix_a[SIZE][SIZE];
    int matrix_b[SIZE][SIZE];
    int result[SIZE][SIZE];

    generate_matrix(matrix_a);
    generate_matrix(matrix_b);

    // Умножение матриц
    multiply_matrices(result, matrix_a, matrix_b);

    // Вывод исходных матриц и результата
    printf("Matrix A:\n");
    print_matrix(matrix_a);

    printf("\nMatrix B:\n");
    print_matrix(matrix_b);

    printf("\nResult:\n");
    print_matrix(result);

    return 0;
}

