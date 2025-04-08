from .KerasModel import KerasModel

class MLP1_2(KerasModel):
    def __init__(self, restaurant_name, name, isVal):
        super().__init__(restaurant_name, name, isVal)

    def get_model_input(self, res_details, diary):
        return res_details.reshape(1,-1)
    
class MLP1(MLP1_2):
    def __init__(self, restaurant_name, isVal):
        super().__init__(restaurant_name, 'MLP1', isVal)

class MLP2(MLP1_2):
    def __init__(self, restaurant_name, isVal):
        super().__init__(restaurant_name, 'MLP2', isVal)