#include <stdio.h>
#include <omp.h>

static long num_steps = 1000000;
double step;

int main() {
    int i;
    double x, pi, sum = 0.0;
    double start_time, end_time;

    step = 1.0 / (double) num_steps;

    // Начало отсчёта времени
    start_time = omp_get_wtime();

    // Распараллеливание цикла с помощью OpenMP
    #pragma omp parallel for reduction(+:sum)
    for (i = 1; i <= num_steps; i++) {
        x = (i - 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    // Конец отсчёта времени
    end_time = omp_get_wtime();

    pi = step * sum;

    // Вывод результатов и времени выполнения
    printf("\n pi with %ld steps is %.10f\n", num_steps, pi);
    printf("Execution time: %f seconds\n", end_time - start_time);

    return 0;
}

