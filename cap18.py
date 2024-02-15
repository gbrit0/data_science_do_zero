# REDES  NEURAIS

# perceptrons
from cap04 import Vector, dot 

def step_function(x: float)->float:
   return 1.0 if x >= 0 else 0.0

def perceptron_output(weights: Vector, bias: float, x: Vector) -> float:
   """Retorna 1 se o perceptron 'disparar', 0 se não"""
   calculation = dot(weights, x) + bias
   return step_function(calculation)


def main():
   and_weights = [2., 2]
   and_bias = -3.

   assert perceptron_output(and_weights, and_bias, [1,1]) == 1
   assert perceptron_output(and_weights, and_bias, [1,0]) == 0
   assert perceptron_output(and_weights, and_bias, [0,1]) == 0
   assert perceptron_output(and_weights, and_bias, [0,0]) == 0

if __name__ == "__main__": main()