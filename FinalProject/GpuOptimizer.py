import time
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class GPUBaseOptimizer:
    def __init__(
        self,
        # Скорость обучения (learning rate). По умолчанию 0.001.
        lr: float = 1e-3,
        # Размер пакета данных для обучения. По умолчанию 4096.
        batch_size: int = 8192,
        epochs: int = 50,  # Количество эпох обучения. По умолчанию 50.
        # Критерий останова: минимальная разница между значениями функции потерь на последовательных эпохах. По умолчанию 1e-6.
        epsilon: float = 1e-6,
    ):
        """
        Инициализация класса оптимизатора для обучения модели на GPU.

        Параметры:
        - lr (float): Скорость обучения (learning rate).
        - batch_size (int): Размер пакета данных для обучения.
        - epochs (int): Количество эпох обучения.
        - epsilon (float): Критерий останова.

        Поля:
        - lr (float): Скорость обучения (learning rate).
        - batch_size (int): Размер пакета данных для обучения.
        - epochs (int): Количество эпох обучения.
        - epsilon (float): Критерий останова.
        - history (dict): История значений функции потерь и параметров на каждой эпохе.
        - device (torch.device): Устройство (CPU или GPU), на котором выполняется вычисление.

        Исключения:
        - ValueError: Если введены некорректные параметры.
        """
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.epsilon = epsilon

        # Словарь для хранения истории значений функции потерь и параметров
        self.history = {"loss": [], "params": []}
        # Определение устройства (GPU или CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Проверка на корректность введенных параметров
        if lr <= 0.0 or batch_size < 0 or epochs < 0 or epsilon < 0.0:
            raise ValueError(f"Введенные параметры некорректны")

    def loss(
        self, X: torch.Tensor, y: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисление функции потерь для заданных данных и параметров модели.

        Параметры:
        - X (torch.Tensor): Матрица признаков.
        - y (torch.Tensor): Вектор целевых значений.
        - params (torch.Tensor): Текущие параметры модели.

        Возвращает:
        - torch.Tensor: Значение функции потерь.

        Примечание:
        Функция использует устройство (GPU или CPU), на котором выполняются вычисления.
        """
        # Перемещение данных на соответствующее устройство (GPU или CPU), если необходимо
        if self.device not in (X.device, y.device, params.device):
            X, y, params = X.to(self.device), y.to(self.device), params.to(self.device)

        # Вычисление ошибки модели и возвращение значения функции потерь
        error = torch.matmul(X, params) - y
        return torch.sum(error**2) / (2 * y.shape[0])

    def update_params(
        self, X: torch.Tensor, y: torch.Tensor, params: torch.Tensor, **kwargs
    ) -> Tensor:
        """
        Обновление параметров модели на основе градиента.

        Параметры:
        - params (torch.Tensor): Текущие параметры модели.
        - gradient (torch.Tensor): Градиент функции потерь по параметрам модели.
        - **kwargs: Дополнительные аргументы, если необходимо.

        Возвращает:
        - Tensor: Обновленные параметры модели.

        Исключения:
        - NotImplementedError: Если метод не реализован в подклассе.
        """
        # Выводим ошибку, если метод не реализован в подклассе
        raise NotImplementedError("Subclasses must implement update_params method.")

    def gradient(
        self, X: torch.Tensor, y: torch.Tensor, params: torch.Tensor
    ) -> Tensor:
        """
        Вычисление градиента функции потерь по параметрам модели.

        Параметры:
        - X (torch.Tensor): Матрица признаков.
        - y (torch.Tensor): Вектор целевых значений.
        - params (torch.Tensor): Текущие параметры модели.

        Возвращает:
        - Tensor: Градиент функции потерь по параметрам модели.

        Примечание:
        Функция использует устройство (GPU или CPU), на котором выполняются вычисления.
        """
        # Перемещение данных на соответствующее устройство (GPU или CPU), если необходимо
        if self.device not in (X.device, y.device, params.device):
            X, y, params = X.to(self.device), y.to(self.device), params.to(self.device)

        # Вычисление ошибки модели и возвращение градиента
        error = torch.matmul(X, params) - y
        return torch.matmul(X.T, error) / y.shape[0]

    def fit(
        self, X: torch.Tensor, y: torch.Tensor, initial_params: torch.Tensor
    ) -> Tensor:
        """
        Обучение модели на заданных данных.

        Параметры:
        - X (torch.Tensor): Матрица признаков.
        - y (torch.Tensor): Вектор целевых значений.
        - initial_params (torch.Tensor): Начальные параметры модели.

        Возвращает:
        - Tensor: Обученные параметры модели.

        Примечание:
        Метод использует устройство (GPU или CPU), на котором выполняются вычисления.
        """
        num_feature = X.shape[1]
        num_samples = X.shape[0]

        if num_feature > 100_000 and num_samples > 20000:
            self.batch_size = 8192


        # Если начальные параметры не заданы, используем случайные значения
        if initial_params is None:
            initial_params = torch.rand(num_feature)

        # Копирование начальных параметров, чтобы не изменять оригинальные значения
        #params = initial_params.clone()
        params = initial_params

        # Перемещение данных на соответствующее устройство (GPU или CPU)
        #X = torch.tensor(X, dtype=torch.float32, device=self.device)
        #y = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        # X = X.to(self.device)
        # y = y.to(self.device)

        # Создание набора данных для DataLoader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        prev_loss = torch.inf  # Предыдущее значение функции потерь
        start_time = time.time()  # Время начала обучения

        # Цикл по эпохам обучения
        for epoch in range(self.epochs):
            epoch_loss = 0  # Суммарная потеря за эпоху
            num_batches = 0  # Количество пакетов данных

            # Цикл по пакетам данных
            for X_batch, y_batch in dataloader:
                # Перемещение пакетов данных на соответствующее устройство (GPU или CPU)
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Вычисление градиента и обновление параметров модели
                #grad = self.gradient(X_batch, y_batch, params)
                params = self.update_params(X_batch, y_batch, params)

                # Вычисление значения функции потерь
                epoch_loss += self.loss(X_batch, y_batch, params)
                num_batches += 1

            epoch_loss /= num_batches  # Усреднение потерь по эпохе

            # Добавление значений функции потерь и параметров в историю
            self.history["loss"].append(epoch_loss)
            self.history["params"].append(params.clone())

            # Проверка на сходимость
            if torch.abs(epoch_loss - prev_loss) < self.epsilon:
                break
            prev_loss = epoch_loss  # Обновление предыдущей потери

        end_time = time.time()  # Время окончания обучения
        time_work = end_time - start_time  # Время затраченного на по
        # Вывод времени, затраченного на поиск оптимальных параметров
        print(
            f"Время, затраченное на поиск оптимальных параметров: {end_time - start_time} секунд"
        )

        return time_work, params, self.history


class GPUMomentum(GPUBaseOptimizer):
    def __init__(self, momentum: float = 0.9, **kwargs):
        """
        Инициализация класса оптимизатора с использованием метода Momentum.

        Параметры:
        - momentum (float): Коэффициент Momentum. По умолчанию 0.9.
        - **kwargs: Дополнительные аргументы, передаваемые в конструктор родительского класса.

        Поля:
        - momentum (float): Коэффициент Momentum.
        - velocity (torch.Tensor): Вектор скорости (градиент Momentum).

        Примечание:
        Данный класс наследует от базового класса GPUBaseOptimizer и переопределяет метод обновления параметров модели.
        """
        super().__init__(**kwargs)
        self.momentum = momentum
        self.velocity = None

    def update_params(
        self, X: torch.Tensor, y: torch.Tensor, params: torch.Tensor, **kwargs
    ) -> Tensor:
        """
        Обновление параметров модели с использованием метода Momentum.

        Параметры:
        - params (torch.Tensor): Текущие параметры модели.
        - gradient (torch.Tensor): Градиент функции потерь по параметрам модели.
        - **kwargs: Дополнительные аргументы, если необходимо.

        Возвращает:
        - Tensor: Обновленные параметры модели.

        Примечание:
        Метод вычисляет градиент Momentum и обновляет параметры модели с использованием скорости обновления.
        """
        # Перемещение параметров на соответствующее устройство (GPU или CPU), если необходимо
        # Перемещение данных на соответствующее устройство (GPU или CPU), если необходимо
        if self.device not in (X.device, y.device, params.device):
            X, y, params = X.to(self.device), y.to(self.device), params.to(self.device)

        # Инициализация вектора скорости (градиента Momentum), если он еще не создан
        if self.velocity is None:
            self.velocity = torch.zeros(params.shape)

        # Перемещение вектора скорости на соответствующее устройство (GPU или CPU)
        self.velocity = self.velocity.to(self.device)

        # Вычисление градиента Momentum и обновление вектора скорости
        self.velocity = self.momentum * self.velocity + self.lr * super().gradient(
            X, y, params
        )

        # Обновление параметров модели с использованием скорости обновления
        return params - self.velocity


class GPURMSprop(GPUBaseOptimizer):
    def __init__(
        self,
        momentum: float = 0.9,
        alpha: float = 0.99,
        eps: float = 1e-8,
        centered: bool = True,
        **kwargs,
    ):
        """
        Инициализация класса оптимизатора с использованием метода RMSprop.

        Параметры:
        - momentum (float): Коэффициент Momentum. По умолчанию 0.9.
        - alpha (float): Параметр сглаживания. По умолчанию 0.99.
        - eps (float): Сглаживающий коэффициент для избежания деления на ноль. По умолчанию 1e-8.
        - centered (bool): Флаг центрирования градиента. По умолчанию True.
        - **kwargs: Дополнительные аргументы, передаваемые в конструктор родительского класса.

        Поля:
        - momentum (float): Коэффициент Momentum.
        - alpha (float): Параметр сглаживания.
        - eps (float): Сглаживающий коэффициент для избежания деления на ноль.
        - centered (bool): Флаг центрирования градиента.
        - velocity (torch.Tensor): Вектор скорости.

        Примечание:
        Данный класс наследует от базового класса GPUBaseOptimizer и переопределяет метод обновления параметров модели.
        """
        super().__init__(**kwargs)

        self.momentum = momentum
        self.alpha = alpha
        self.centered = centered
        self.eps = eps
        self.velocity = None

    def update_params(
        self, X: torch.Tensor, y: torch.Tensor, params: torch.Tensor, **kwargs
    ):
        """
        Обновление параметров модели с использованием метода RMSprop.

        Параметры:
        - params (torch.Tensor): Текущие параметры модели.
        - gradient (torch.Tensor): Градиент функции потерь по параметрам модели.
        - **kwargs: Дополнительные аргументы, если необходимо.

        Возвращает:
        - torch.Tensor: Обновленные параметры модели.

        Примечание:
        Метод вычисляет градиент и обновляет параметры модели с использованием метода RMSprop.
        """
        # Перемещение данных на соответствующее устройство (GPU или CPU), если необходимо
        if self.device not in (X.device, y.device, params.device):
            X, y, params = X.to(self.device), y.to(self.device), params.to(self.device)

        # Инициализация вектора скорости (градиента)
        if self.velocity is None:
            self.velocity = torch.zeros(params.shape)

        # Перемещение вектора скорости на соответствующее устройство (GPU или CPU)
        self.velocity = self.velocity.to(self.device)

        # Обновление вектора скорости (градиента) с использованием метода RMSprop
        self.velocity = (
            self.alpha
            + self.velocity
            + (1 - self.alpha) * (super().gradient(X, y, params)) ** 2
        )

        # Обновление параметров модели с использованием метода RMSprop
        return params - self.lr / (torch.sqrt(self.velocity) + self.eps) * (
            super().gradient(X, y, params)
        )


class GPUAdam(GPUBaseOptimizer):
    def __init__(
        self, betta1: float = 0.9, betta2: float = 0.999, eps: float = 1e-8, **kwargs
    ):
        """
        Инициализация класса оптимизатора с использованием метода Adam.

        Параметры:
        - betta1 (float): Коэффициент для оценки первого момента (moment1). По умолчанию 0.9.
        - betta2 (float): Коэффициент для оценки второго момента (moment2). По умолчанию 0.999.
        - eps (float): Сглаживающий коэффициент для избежания деления на ноль. По умолчанию 1e-8.
        - **kwargs: Дополнительные аргументы, передаваемые в конструктор родительского класса.

        Поля:
        - betta1 (float): Коэффициент для оценки первого момента (moment1).
        - betta2 (float): Коэффициент для оценки второго момента (moment2).
        - eps (float): Сглаживающий коэффициент для избежания деления на ноль.
        - moment1 (torch.Tensor): Первый момент.
        - moment2 (torch.Tensor): Второй момент.

        Примечание:
        Данный класс наследует от базового класса GPUBaseOptimizer и переопределяет метод обновления параметров модели.
        """
        super().__init__(**kwargs)
        self.betta1 = betta1
        self.betta2 = betta2
        self.eps = eps
        self.moment1 = None
        self.moment2 = None

    def update_params(
        self, X: torch.Tensor, y: torch.Tensor, params: torch.Tensor, t: int, **kwargs
    ) -> Tensor:
        """
        Обновление параметров модели с использованием метода Adam.

        Параметры:
        - params (torch.Tensor): Текущие параметры модели.
        - gradient (torch.Tensor): Градиент функции потерь по параметрам модели.
        - t (int): Текущий номер шага (итерации).
        - **kwargs: Дополнительные аргументы, если необходимо.

        Возвращает:
        - torch.Tensor: Обновленные параметры модели.

        Примечание:
        Метод вычисляет градиент и обновляет параметры модели с использованием метода Adam.
        """
        # Перемещение данных на соответствующее устройство (GPU или CPU), если необходимо
        # Перемещение данных на соответствующее устройство (GPU или CPU), если необходимо
        if self.device not in (X.device, y.device, params.device):
            X, y, params = X.to(self.device), y.to(self.device), params.to(self.device)

        # Инициализация первого и второго моментов, если они еще не созданы
        if self.moment1 is None:
            self.moment1 = torch.zeros(params.shape)
            self.moment1 = self.moment1.to(self.device)

        if self.moment2 is None:
            self.moment2 = torch.zeros(params.shape)
            self.moment2 = self.moment2.to(self.device)

        # Обновление первого момента (moment1) и второго момента (moment2) с использованием метода Adam
        self.moment1 = self.betta1 * self.moment1 + (1 - self.betta1) * super().gradient(
            X, y, params
        )
        self.moment2 = self.betta2 * self.moment2 + (1 - self.betta2) * (
            super().gradient(X, y, params) ** 2
        )

        # Корректировка моментов для учета bias
        moment1_corrected = self.moment1 / (1 - self.betta1**t)
        moment2_corrected = self.moment2 / (1 - self.betta2**t)

        # Обновление параметров модели с использованием метода Adam
        if torch.any(torch.abs(moment1_corrected) < self.epsilon) or torch.any(
            torch.abs(moment2_corrected) < self.epsilon
        ):
            updated_params = params
        else:
            updated_params = params - self.lr * moment1_corrected / (
                torch.sqrt(moment2_corrected) + self.epsilon
            )

        return updated_params

    def fit(
        self, X: torch.Tensor, y: torch.Tensor, initial_params: torch.Tensor
    ) -> Tensor:
        """
        Обучение модели на заданных данных с использованием оптимизатора.

        Параметры:
        - X (torch.Tensor): Матрица признаков.
        - y (torch.Tensor): Вектор целевых значений.
        - initial_params (torch.Tensor): Начальные параметры модели.

        Возвращает:
        - Tensor: Обновленные параметры модели.

        Примечание:
        Метод использует устройство (GPU или CPU) для выполнения вычислений.
        """
        num_feature = X.shape[1]    # Количество векторов
        num_samples = X.shape[0]    # Количество примеров

        # Если начальные параметры не заданы, используем случайные значения
        if initial_params is None:
            initial_params = torch.rand(num_feature)

        if num_feature > 100_000 and num_samples > 20000:
            self.batch_size = 8192

        params = initial_params.clone()
        params = params.to(self.device)
        t = 0

        # Перемещение данных на соответствующее устройство (GPU или CPU)
        # X = torch.tensor(X, dtype=torch.float32, device=self.device)
        # y = torch.tensor(y, dtype=torch.float32, device=self.device)

        # X = X.to(self.device)
        # y = y.to(self.device)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        prev_loss = torch.inf
        start_time = time.time()

        # Цикл по эпохам обучения
        for epoch in range(self.epochs):
            epoch_loss = 0
            num_batches = 0

            # Цикл по пакетам данных
            for X_batch, y_batch in dataloader:
                t += 1

                # Перемещение пакетов данных на соответствующее устройство (GPU или CPU)
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Вычисление градиента и обновление параметров модели с использованием текущего оптимизатора
                #grad = self.gradient(X_batch, y_batch, params)
                params = self.update_params(X_batch, y_batch, params, t)

                # Вычисление значения функции потерь и обновление счетчика пакетов данных
                epoch_loss += self.loss(X_batch, y_batch, params)
                num_batches += 1

            epoch_loss /= num_batches  # Усреднение потерь по эпохе

            # Добавление значения функции потерь и параметров в историю
            self.history["loss"].append(epoch_loss)
            self.history["params"].append(params.clone())

            #prev_loss = epoch_loss
            # Проверка на сходимость и обновление предыдущей потери
            if torch.abs(epoch_loss - prev_loss) < self.epsilon:
                print(f"Converged after epoch {epoch}.")
                break
            prev_loss = epoch_loss

        end_time = time.time()  # Время окончания обучения

        # Вывод времени, затраченного на поиск оптимальных параметров
        print(
            f"Время, затраченное на поиск оптимальных параметров: {end_time-start_time} секунд"
        )

        time_work = end_time - start_time  # Время затраченного на по
        return time_work, params, self.history
