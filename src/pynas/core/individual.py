from . import architecture_builder as builder
from ..train import myFit
from copy import deepcopy


evaluator = myFit.FitnessEvaluator()

class Individual:
    """
    The `Individual` class represents an individual entity in the genetic algorithm or evolutionary computation context. 
    It encapsulates the architecture, chromosome, and associated properties such as fitness, IOU (Intersection over Union), 
    FPS (Frames Per Second), and model size. The class provides methods for converting between architecture and chromosome 
    representations, resetting properties, and creating deep copies of the individual.
    Attributes:
        architecture (str): The architecture code representing the individual's structure.
        chromosome (list): A list representation of the architecture code.
        parsed_layers (list): Parsed layers of the architecture code.
        fitness (float): The fitness score of the individual.
        iou (float or None): The Intersection over Union metric.
        metric (float or None): Currently stores IoU (Intersection over Union).

        fps (float or None): The Frames Per Second metric.
        model_size (float or None): The size of the model.
        model (object or None): The trained model associated with the individual.
    Methods:
        __init__(max_layers, min_layers=3):
            Initializes an individual with a random architecture and its corresponding chromosome.
        __str__():
            Returns a string representation of the individual.
        _reparse_layers():
            Reparses the layers of the architecture based on the chromosome.
        reset():
        architecture2chromosome(input_architecture):
            Converts an architecture code into a chromosome list.
        chromosome2architecture(input_chromosome):
            Converts a chromosome list back into an architecture code.
        copy():
            Creates a deep copy of the individual, including its properties and model.
        set_trained_model(model):
            Sets the trained model for the individual.
    """
    def __init__(self, max_layers, min_layers=3):
        self.architecture = builder.generate_random_architecture_code(max_layers=max_layers, min_layers=min_layers)
        self.chromosome = self.architecture2chromosome(input_architecture=self.architecture)
        self.parsed_layers = builder.parse_architecture_code(self.architecture)
        self.reset()

    def __str__(self):
        return f'Individual: {self.architecture}'
    
    
    def _reparse_layers(self):
        self.parsed_layers = builder.parse_architecture_code(self.chromosome2architecture(self.chromosome))


    def reset(self):
        """
        Resets the individual's fitness, IOU, FPS, and model size.
        """
        self.results = {}
        self.fitness = 0.0
        self.metric = None
        self.fps = None
        self.model_size = None
        self.model = None


    # Implement the logic to prompt the fitnes
    def _prompt_fitness(self):
        # fps = results['fps']
        #metric = results['test_mcc']
        # metric = results['test_iou']
        # self.fps, self.metric = fps, metric
        # self.results = results

        self.fitness = evaluator.weighted_sum_exponential(self.fps, self.iou)

        print("IoU:", self.iou)
        print("FPS:", self.fps)
        print("Fitness:", self.fitness)
        return self.fitness

    def architecture2chromosome(self, input_architecture):
        """
        Converts an architecture code into a chromosome list by splitting
        the architecture code using 'E'. This method also handles the case where
        the architecture ends with 'EE', avoiding an empty string at the end of the list.
        """
        # Split the architecture code on 'E'
        chromosome = input_architecture.split('E')
        # Remove the last two empty elements if the architecture ends with 'EE'
        if len(chromosome) >= 2 and chromosome[-1] == '' and chromosome[-2] == '':
            chromosome = chromosome[:-2]
        elif len(chromosome) >= 1 and chromosome[-1] == '':
            # If it only ends with a single 'E', just remove the last empty element
            chromosome = chromosome[:-1]
        return chromosome


    def chromosome2architecture(self, input_chromosome):
        """
        Converts the chromosome list back into an architecture code by joining
        the list items with 'E' and ensuring the architecture ends with 'EE'.
        """
        architecture_code = 'E'.join(input_chromosome) + 'EE'
        return architecture_code


    def copy(self):
        """
        Creates a deep copy of the current individual, including architecture,
        chromosome, and fitness.
        """
        new_individual = Individual(max_layers=len(self.chromosome))
        new_individual.architecture = deepcopy(self.architecture)
        new_individual.chromosome = deepcopy(self.chromosome)
        new_individual.fitness = self.fitness
        new_individual.iou = self.iou
        new_individual.fps = self.fps
        new_individual.model_size = self.model_size
        
        if self.model is not None:
            new_individual.model = deepcopy(self.model)  # Copy the entire model

        return new_individual    

    
    def set_trained_model(self, model):
        """
        Set the trained model.
        """
        self.model = model

