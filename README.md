# genetic_model
Its a class that runs the genetic algorithm and its writen alike keras api models.
The example that I ran the genetic algorithm on is an example of binary parameters optimization that I found in one of my AI class exersices in university.
# Document
You can create a model with code ```python
model = GeneticAlgorithmModel(gene_size, population_size)
```
#
Then you feed functions with model.compile(crossover_fun, mutation_fun, random_population_fun, crossover_coeff, mutation_coeff)
#
Then you use r = model.fit(epochs, metrics=[]) to train the model
#
You can use r list to show what happend!
