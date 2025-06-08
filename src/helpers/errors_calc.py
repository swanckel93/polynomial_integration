class ErrorCalculations:
    @staticmethod
    def get_relative_error(target_value: float, current_value: float) -> float:
        assert target_value is not None
        return abs(current_value - target_value) / target_value * 100

    @staticmethod
    def get_absolute_error(target_value: float, current_value: float) -> float:
        assert target_value is not None
        return abs(target_value - current_value)
