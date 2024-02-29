import time
import numpy as np


class BaseOptimizer:
    """
    Базовый класс для реализации методов оптимизации.
    """

    def __init__(self, lr: float = 1e-3, batch_size: int = 64, epochs: int = 50, epsilon: float = 1e-6):
        """
        Конструктор класса BaseOptimizer.

        Args:
        - lr: float, скорость обучения (learning rate).
        - batch_size: int, размер батча для обучения.
        - epochs: int, количество эпох обучения.
        - epsilon: float, значение для предотвращения деления на ноль.

        Примечание:
        Инициализирует экземпляр базового класса метода оптимизации с заданными параметрами.
        """
        self.lr = lr  # Скорость обучения
        self.batch_size = batch_size  # Размер батча
        self.epochs = epochs  # Количество эпох
        self.epsilon = epsilon  # Значение для предотвращения деления на ноль
        # История обучения (значения функции потерь и параметров модели)
        self.history = {'loss': [], 'params': []}

        # Проверка на корректность введенных параметров
        if lr <= 0.0 or batch_size < 0 or epochs < 0 or epsilon < 0.0:
            raise ValueError(f'Введенные параметры некорректны')
    # функция потерь MSE

    def loss(self, X, y, params) -> float:
        """
        Метод для вычисления функции потерь Mean Squared Error (MSE).

        Args:
        - X: numpy.ndarray, входные признаки размерности (n_samples, n_features),
             где n_samples - количество образцов, n_features - количество признаков.
        - y: numpy.ndarray, метки размерности (n_samples,), где n_samples - количество образцов.
        - params: numpy.ndarray, веса модели.

        Returns:
        - Значение функции потерь MSE.

        Примечание:
        Метод вычисляет функцию потерь MSE для данных X, меток y и параметров модели params.
        """

        error = np.dot(X, params) - y
        return sum(error**2) / (2*len(y))

    # подсчет градиента(в точке) для функции потерь MSE
    def gradient(self, X, y, params) -> np.ndarray:
        """
        Метод для вычисления градиента функции потерь Mean Squared Error (MSE).

        Args:
        - X: numpy.ndarray, входные признаки размерности (n_samples, n_features),
             где n_samples - количество образцов, n_features - количество признаков.
        - y: numpy.ndarray, метки размерности (n_samples,), где n_samples - количество образцов.
        - params: numpy.ndarray, веса модели.

        Returns:
        - Градиент функции потерь MSE по параметрам модели.

        Примечание:
        Метод вычисляет градиент функции потерь MSE для данных X, меток y и параметров модели params.
        """
        # return np.dot(X.T, (np.dot(X, params)-y))
        error = np.dot(X, params) - y
        return np.dot(X.T, error) / len(y)

    # обновления весов
    def update_params(self, params, gradient):
        """
        Метод для обновления параметров модели.

        Args:
        - params: numpy.ndarray, текущие параметры модели.
        - gradient: numpy.ndarray, градиент функции потерь по параметрам модели.

        Returns:
        - Новые параметры модели.

        Примечание:
        Этот метод должен быть переопределен в дочерних классах, чтобы реализовать конкретный метод обновления весов.
        """
        raise NotImplementedError(
            "Subclasses must implement update_params method.")

    # генерация батчей
    def generate_batches(self, X, y):
        """
        Метод для генерации батчей данных из входных признаков X и меток y.

        Args:
        - X: numpy.ndarray, входные признаки размерности (n_samples, n_features),
             где n_samples - количество образцов, n_features - количество признаков.
        - y: numpy.ndarray, метки размерности (n_samples,), где n_samples - количество образцов.

        Returns:
        - Генератор, который возвращает батчи данных в каждой итерации.
          Каждый батч содержит X_batch и y_batch, соответствующие выбранным индексам.

        Примечание:
        Метод использует случайное перемешивание индексов образцов для получения различных батчей в каждой эпохе обучения.
        """
        num_samples = X.shape[0]  # Определение общего количества образцов
        indices = np.arange(num_samples)  # Создание массива индексов образцов
        np.random.shuffle(indices)  # Перемешивание индексов

        # Итерация по индексам для создания батчей
        for start_idx in range(0, num_samples, self.batch_size):
            # Вычисление конечного индекса батча
            end_idx = min(start_idx + self.batch_size, num_samples)
            # Выбор индексов для текущего батча
            batch_indices = indices[start_idx:end_idx]
            # Возврат батча данных X и соответствующих меток y
            yield X[batch_indices], y[batch_indices]

    def fit(self, X, y, initial_params):
        """
        Метод для обучения модели(весов).

        Args:
        - X: numpy.ndarray, входные признаки размерности (n_samples, n_features),
             где n_samples - количество образцов, n_features - количество признаков.
        - y: numpy.ndarray, метки размерности (n_samples,), где n_samples - количество образцов.
        - initial_params: numpy.ndarray, начальные параметры модели.

        Returns:
        - Обученные параметры модели.

        Примечание:
        Метод обучает модель с использованием градиентного спуска по эпохам. На каждой эпохе происходит обход всех батчей,
        где для каждого батча вычисляется градиент и обновляются параметры модели. После завершения эпохи вычисляется средняя
        потеря для всех батчей, и если изменение этой потери меньше, чем заданное значение epsilon, обучение считается завершенным.
        Все потери и параметры модели сохраняются в истории для дальнейшего анализа.
        """
        num_feature = X.shape[1]

        if initial_params is None:
            initial_params = np.random.uniform(size=num_feature)

        # Создание копии начальных параметров
        params = initial_params.copy()
        prev_loss = np.inf  # Инициализация предыдущей потери как бесконечности

        start_time = time.time()

        # Итерация по эпохам
        for epoch in range(self.epochs):

            epoch_loss = 0  # Инициализация суммарной потери на текущей эпохе
            num_batches = 0  # Инициализация счетчика количества батчей на текущей эпохе

            # Итерация по батчам данных
            for X_batch, y_batch in self.generate_batches(X, y):

                # Вычисление градиента
                grad = self.gradient(X_batch, y_batch, params)
                # Обновление параметров модели
                params = self.update_params(X_batch, y_batch, params)
                # Вычисление потери для текущего батча
                epoch_loss += self.loss(X_batch, y_batch, params)
                num_batches += 1  # Увеличение счетчика батчей

                epoch_loss /= num_batches  # Вычисление средней потери на текущей эпохе
                # Сохранение потери в истории
                self.history['loss'].append(epoch_loss)
                # Сохранение весов в истории
                self.history['params'].append(params.copy())

                # Проверка сходимости: если изменение потери меньше заданного значения tol, обучение считается завершенным
                if np.abs(epoch_loss - prev_loss) < self.epsilon:
                    print(f"Converged after epoch {epoch}.")
                    break

                prev_loss = epoch_loss  # Обновление предыдущей потери

        end_time = time.time()
        print(f'Time spent searching for optimal parameters {end_time-start_time} seconds')
        time_work = end_time - start_time  # Время затраченного на по

        return time_work, params, self.history  # Возвращение обученных параметров модели


class Momentum(BaseOptimizer):
    """
    Класс, реализующий метод оптимизации Momentum для обновления параметров модели.
    """

    def __init__(self, momentum: float = 0.9, lr: float = 2, **kwargs):
        """
        Конструктор класса Momentum.

        Args:
        - momentum: float, параметр инерции, который указывает на то, как сильно следует использовать
                    предыдущие изменения весов при обновлении.
        - lr: float, скорость обучения (learning rate).
        - **kwargs: дополнительные аргументы, передаваемые в родительский класс BaseOptimizer.

        Примечание:
        Инициализирует экземпляр класса метода оптимизации Momentum с заданными параметрами.
        """
        super().__init__(
            **
            kwargs)  # Вызываем конструктор родительского класса BaseOptimizer
        self.momentum = momentum  # Коэффициент инерции
        self.velocity = None  # Скорость обновления параметров

    def update_params(self, X, y, params):
        """
        Метод для обновления параметров модели с использованием метода оптимизации Momentum.

        Args:
        - params: numpy.ndarray, текущие параметры модели.
        - gradient: numpy.ndarray, градиент функции потерь по параметрам модели.

        Returns:
        - Новые параметры модели после обновления.

        Примечание:
        Метод рассчитывает новые параметры модели с использованием метода оптимизации Momentum.
        """
        if self.velocity is None:
            self.velocity = np.random.uniform(
                size=params.shape
            )  # Инициализация скорости случайными значениями

        self.velocity = self.momentum * self.velocity + self.lr * super(
        ).gradient(X, y, params)  # Обновление скорости с учетом инерции

        return params - self.velocity  # Возвращение новых параметров модели после обновления


class RMSprop(BaseOptimizer):

    def __init__(self,
                 momentum: float = 0.9,
                 alpha: float = 0.99,
                 eps: float = 1e-8,
                 centered: bool = True,
                 **kwargs):

        super().__init__(**kwargs)

        self.momentum = momentum
        self.alpha = alpha
        self.eps = eps
        self.centered = centered
        self.velocity = None

    def update_params(self, X, y, params):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        self.velocity = self.alpha + self.velocity + (1 - self.alpha) * (
            super().gradient(X, y, params))**2

        return params - self.lr / (np.sqrt(self.velocity) +
                                   self.eps) * (super().gradient(X, y, params))


class Adam(BaseOptimizer):
    """
    Класс реализующий метод оптимизации Adam для обновления параметров модели.
    """

    def __init__(self,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8,
                 **kwargs):
        """
        Конструктор класса Adam.

        Args:
        - beta1: float, коэффициент сглаживания для первого момента.
        - beta2: float, коэффициент сглаживания для второго момента.
        - epsilon: float, значение для предотвращения деления на ноль.
        - **kwargs: дополнительные аргументы, передаваемые в родительский класс BaseOptimizer.

        Примечание:
        Инициализирует экземпляр класса метода оптимизации Adam с заданными параметрами.
        """
        super().__init__(
            **
            kwargs)  # Вызываем конструктор родительского класса BaseOptimizer
        self.beta1 = beta1  # Коэффициент сглаживания для первого момента
        self.beta2 = beta2  # Коэффициент сглаживания для второго момента
        self.eps = eps  # Значение для предотвращения деления на ноль
        self.moment1 = None  # Первый момент
        self.moment2 = None  # Второй момент

    def update_params(self, X, y, params, t):
        """
        Метод для обновления параметров модели с использованием метода оптимизации Adam.

        Args:
        - params: numpy.ndarray, текущие параметры модели.
        - gradient: numpy.ndarray, градиент функции потерь по параметрам модели.

        Returns:
        - Новые параметры модели после обновления.

        Примечание:
        Метод рассчитывает обновленные параметры модели с использованием метода оптимизации Adam.
        """

        if self.moment1 is None:
            self.moment1 = np.zeros_like(
                params)  # Инициализируем первый момент нулевым вектором
        if self.moment2 is None:
            self.moment2 = np.zeros_like(
                params)  # Инициализируем второй момент нулевым вектором

        # Обновляем первый и второй моменты
        self.moment1 = self.beta1 * self.moment1 + (
            1 - self.beta1) * super().gradient(X, y, params)
        self.moment2 = self.beta2 * self.moment2 + (1 - self.beta2) * (
            super().gradient(X, y, params)**2)

        # Корректировка моментов для bias
        moment1_corrected = self.moment1 / (1 - self.beta1**t)
        moment2_corrected = self.moment2 / (1 - self.beta2**t)

        if np.any(np.abs(moment1_corrected) < self.epsilon) or np.any(
                np.abs(moment2_corrected) < self.epsilon):
            # Оставляем параметры без изменения, чтобы избежать деления на ноль
            updated_params = params
        else:
            # Обновляем параметры модели
            updated_params = params - self.lr * moment1_corrected / (
                np.sqrt(moment2_corrected) + self.epsilon)

        return updated_params

    def fit(self, X, y, initial_params):
        """
        Метод для обучения модели(весов).

        Args:
        - X: numpy.ndarray, входные признаки размерности (n_samples, n_features),
             где n_samples - количество образцов, n_features - количество признаков.
        - y: numpy.ndarray, метки размерности (n_samples,), где n_samples - количество образцов.
        - initial_params: numpy.ndarray, начальные параметры модели.

        Returns:
        - Обученные параметры модели.

        Примечание:
        Метод обучает модель с использованием градиентного спуска по эпохам. На каждой эпохе происходит обход всех батчей,
        где для каждого батча вычисляется градиент и обновляются параметры модели. После завершения эпохи вычисляется средняя
        потеря для всех батчей, и если изменение этой потери меньше, чем заданное значение epsilon, обучение считается завершенным.
        Все потери и параметры модели сохраняются в истории для дальнейшего анализа.
        """
        num_feature = X.shape[1]

        if initial_params is None:
            initial_params = np.random.uniform(size=num_feature)

        params = initial_params.copy()  # Создание копии начальных параметров
        prev_loss = np.inf  # Инициализация предыдущей потери как бесконечности
        t = 0  # Инициализация колчества итерации

        start_time = time.time()

        # Итерация по эпохам
        for epoch in range(self.epochs):

            epoch_loss = 0  # Инициализация суммарной потери на текущей эпохе
            num_batches = 0  # Инициализация счетчика количества батчей на текущей эпохе

            # Итерация по батчам данных
            for X_batch, y_batch in self.generate_batches(X, y):

                t += 1  # Увеличиваем итерацию

                grad = self.gradient(X_batch, y_batch,
                                     params)  # Вычисление градиента
                params = self.update_params(X, y, params,
                                            t)  # Обновление параметров модели
                epoch_loss += self.loss(
                    X_batch, y_batch,
                    params)  # Вычисление потери для текущего батча
                num_batches += 1  # Увеличение счетчика батчей

                epoch_loss /= num_batches  # Вычисление средней потери на текущей эпохе
                self.history['loss'].append(
                    epoch_loss)  # Сохранение потери в истории
                self.history['params'].append(
                    params.copy())  # Сохранение весов в истории

                # Проверка сходимости: если изменение потери меньше заданного значения tol, обучение считается завершенным
                if np.abs(epoch_loss - prev_loss) < self.epsilon:
                    print(f"Converged after epoch {epoch}.")
                    break

                prev_loss = epoch_loss  # Обновление предыдущей потери

        end_time = time.time()
        print(
            f'Time spent searching for optimal parameters {end_time - start_time} seconds')
        time_work = end_time - start_time  # Время затраченного на по

        return time_work, params, self.history  # Возвращение обученных параметров модели
