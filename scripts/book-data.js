# Step 1: Adding the "AI Book" content to the knowledge base

# Generative AI: Concepts and Fundamentals
add_knowledge("""
Generative AI is a branch of artificial intelligence that focuses on creating new data that resembles existing data, like text, images, or even sound. It uses machine learning models, primarily neural networks, to generate data that can be creative or informative.
""")
add_knowledge("""
Prompt engineering is the process of designing inputs for generative models to produce specific outputs. By carefully crafting prompts, users can guide models to create responses that are relevant and accurate. Effective prompts clarify context, specify details, and may ask questions to elicit structured responses.
""")
add_knowledge("""
Training generative models involves feeding them large datasets and adjusting internal parameters so they can understand patterns in the data. Fine-tuning is an additional step where a pre-trained model is refined on a specific dataset to specialize it for a particular domain or style, like legal text or scientific articles.
""")
add_knowledge("""
RLHF is a technique where a model learns to improve responses by receiving feedback from humans. The model is rewarded for producing high-quality answers, encouraging it to generate better responses over time.
""")

# Machine Learning: Core Concepts and Algorithms
add_knowledge("""
Machine learning is a subset of AI where computers learn from data to make predictions or decisions. Rather than being explicitly programmed, they improve over time by identifying patterns in data.
""")
add_knowledge("""
In supervised learning, the model is trained on labeled data, where each example has an input and an expected output. The model learns to map inputs to outputs by minimizing errors in predictions. Common algorithms include linear regression, decision trees, and support vector machines.
""")
add_knowledge("""
Unsupervised learning models work with unlabeled data, trying to discover hidden patterns or groupings within the data. Clustering (e.g., k-means) and dimensionality reduction (e.g., PCA) are popular techniques.
""")
add_knowledge("""
Overfitting happens when a model learns too much from the training data, capturing noise rather than the underlying pattern. Underfitting occurs when the model fails to capture enough detail in the data. Balancing model complexity is crucial for optimal performance.
""")
add_knowledge("""
Common metrics for evaluating models include accuracy, precision, recall, and F1 score. Regression models may use metrics like RMSE (Root Mean Square Error) or MAE (Mean Absolute Error). These metrics help determine a model's accuracy and reliability.
""")

# NLP: Understanding Text and Language
add_knowledge("""
Natural Language Processing is a field of AI that enables computers to understand, interpret, and respond to human language. NLP powers applications like translation, chatbots, and voice assistants.
""")
add_knowledge("""
Tokenization is the process of breaking text into individual units, like words or subwords. Embeddings convert these tokens into numerical vectors, capturing semantic meaning. Word embeddings (e.g., Word2Vec) allow models to understand relationships between words.
""")
add_knowledge("""
Transformers revolutionized NLP by enabling models to pay attention to different parts of a sentence, helping them capture context better. Attention mechanisms assign weight to words based on their relevance to other words in a sentence, improving comprehension.
""")
add_knowledge("""
Question-answering models use NLP techniques to interpret questions and retrieve relevant information from a dataset. By understanding context, these models provide precise answers based on the question's intent.
""")

# Neural Networks and Deep Learning
add_knowledge("""
Neural networks are computing systems inspired by the human brain. They consist of interconnected nodes (neurons) organized into layers. Networks learn by adjusting the connections (weights) between neurons based on training data.
""")
add_knowledge("""
Types of neural networks include feedforward neural networks, convolutional neural networks (CNNs) for image processing, recurrent neural networks (RNNs) for sequential data, and transformers specialized for NLP.
""")
add_knowledge("""
Backpropagation is the process of adjusting weights in the network by calculating the error in predictions. Gradient descent is an optimization algorithm used to minimize this error, adjusting weights iteratively until the model learns the data patterns.
""")
add_knowledge("""
Activation functions introduce non-linearities into neural networks, enabling them to learn complex patterns. Common activation functions include ReLU, sigmoid, and tanh.
""")

# Ethics in AI: Building Responsible and Transparent AI Systems
add_knowledge("""
AI models can inadvertently learn biases present in training data, leading to unfair or discriminatory outcomes. Ensuring data diversity, reviewing model outputs, and adjusting parameters help create fairer AI systems.
""")
add_knowledge("""
Explainable AI (XAI) focuses on making AI decisions understandable to humans. Models need to be interpretable, especially in high-stakes areas like healthcare or finance, where understanding the "why" behind a decision is essential.
""")
add_knowledge("""
Transparency in AI requires clear documentation of how models are developed, trained, and used. Accountability ensures that developers and organizations take responsibility for their AI models' impacts.
""")
add_knowledge("""
What is Machine Learning
Machine learning is a subfield of computer science that is concerned with building algorithms
which, to be useful, rely on a collection of examples of some phenomenon. These examples
can come from nature, be handcrafted by humans or generated by another algorithm.
Machine learning can also be defined as the process of solving a practical problem by 1)
gathering a dataset, and 2) algorithmically building a statistical model based on that dataset.
That statistical model is assumed to be used somehow to solve the practical problem.
To save keystrokes, I use the terms “learning” and “machine learning” interchangeably. """)
add_knowledge("""Types of Learning
Learning can be supervised, semi-supervised, unsupervised and reinforcement.Supervised Learning
In supervised learning1, the dataset is the collection of labeled examples {(xi, yi)}N
i=1.
Each element xi among N is called a feature vector. A feature vector is a vector in which
each dimension j = 1, . . . , D contains a value that describes the example somehow. That
value is called a feature and is denoted as x(j). For instance, if each example x in our
collection represents a person, then the first feature, x(1), could contain height in cm, the
second feature, x(2), could contain weight in kg, x(3) could contain gender, and so on. For all
examples in the dataset, the feature at position j in the feature vector always contains the
same kind of information. It means that if x(2)
i contains weight in kg in some example xi,
then x(2)
k will also contain weight in kg in every example xk, k = 1, . . . , N . The label yi can
be either an element belonging to a finite set of classes {1, 2, . . . , C}, or a real number, or a
more complex structure, like a vector, a matrix, a tree, or a graph. Unless otherwise stated,
in this book yi is either one of a finite set of classes or a real number. You can see a class as
a category to which an example belongs. For instance, if your examples are email messages
and your problem is spam detection, then you have two classes {spam, not_spam}.
The goal of a supervised learning algorithm is to use the dataset to produce a model
that takes a feature vector x as input and outputs information that allows deducing the label
for this feature vector. For instance, the model created using the dataset of people could
take as input a feature vector describing a person and output a probability that the person
has cancer """)
add_knowledge("""Unsupervised Learning
In unsupervised learning, the dataset is a collection of unlabeled examples {xi}N
i=1.
Again, x is a feature vector, and the goal of an unsupervised learning algorithm is
to create a model that takes a feature vector x as input and either transforms it into
another vector or into a value that can be used to solve a practical problem. For example,
in clustering, the model returns the id of the cluster for each feature vector in the dataset.
In dimensionality reduction, the output of the model is a feature vector that has fewer
features than the input x; in outlier detection, the output is a real number that indicates
how x is dierent from a “typical” example in the dataset """)
add_knowledge("""1.2.2 Unsupervised Learning
In unsupervised learning, the dataset is a collection of unlabeled examples {xi}N
i=1.
Again, x is a feature vector, and the goal of an unsupervised learning algorithm is
to create a model that takes a feature vector x as input and either transforms it into
another vector or into a value that can be used to solve a practical problem. For example,
in clustering, the model returns the id of the cluster for each feature vector in the dataset.
In dimensionality reduction, the output of the model is a feature vector that has fewer
features than the input x; in outlier detection, the output is a real number that indicates
how x is dierent from a “typical” example in the dataset.
1.2.3 Semi-Supervised Learning
In semi-supervised learning, the dataset contains both labeled and unlabeled examples.
Usually, the quantity of unlabeled examples is much higher than the number of labeled
examples. The goal of a semi-supervised learning algorithm is the same as the goal of
the supervised learning algorithm. The hope here is that using many unlabeled examples can
help the learning algorithm to find (we might say “produce” or “compute”) a better model2.
1.2.4 Reinforcement Learning
Reinforcement learning is a subfield of machine learning where the machine “lives” in an
environment and is capable of perceiving the state of that environment as a vector of
features. The machine can execute actions in every state. Dierent actions bring dierent
rewards and could also move the machine to another state of the environment. The goal
of a reinforcement learning algorithm is to learn a policy. A policy is a function f (similar
to the model in supervised learning) that takes the feature vector of a state as input and
outputs an optimal action to execute in that state. The action is optimal if it maximizes the
expected average reward. """)
add_knowledge("""How Supervised Learning Works
In this section, I briefly explain how supervised learning works so that you have the picture
of the whole process before we go into detail. I decided to use supervised learning as an
example because it’s the type of machine learning most frequently used in practice.
The supervised learning process starts with gathering the data. The data for supervised
learning is a collection of pairs (input, output). Input could be anything, for example, email
messages, pictures, or sensor measurements. Outputs are usually real numbers, or labels (e.g.
“spam”, “not_spam”, “cat”, “dog”, “mouse”, etc). In some cases, outputs are vectors (e.g.,
four coordinates of the rectangle around a person on the picture), sequences (e.g. [“adjective”,
“adjective”, “noun”] for the input “big beautiful car”), or have some other structure.
Let’s say the problem that you want to solve using supervised learning is spam detection.
You gather the data, for example, 10,000 email messages, each with a label either “spam” or
“not_spam” (you could add those labels manually or pay someone to do that for us). Now,
you have to convert each email message into a feature vector.
The data analyst decides, based on their experience, how to convert a real-world entity, such
as an email message, into a feature vector. One common way to convert a text into a feature
vector, called bag of words, is to take a dictionary of English words (let’s say it contains
20,000 alphabetically sorted words) and stipulate that in our feature vector:
• the first feature is equal to 1 if the email message contains the word “a”; otherwise,
this feature is 0;
• the second feature is equal to 1 if the email message contains the word “aaron”; otherwise,
this feature equals 0;
• . . .
• the feature at position 20,000 is equal to 1 if the email message contains the word
“zulu”; otherwise, this feature is equal to 0.
You repeat the above procedure for every email message in our collection, which gives
us 10,000 feature vectors (each vector having the dimensionality of 20,000) and a label
(“spam”/“not_spam”).
Now you have a machine-readable input data, but the output labels are still in the form of
human-readable text. Some learning algorithms require transforming labels into numbers.
For example, some algorithms require numbers like 0 (to represent the label “not_spam”)
and 1 (to represent the label “spam”). The algorithm I use to illustrate supervised learning is
called Support Vector Machine (SVM). This algorithm requires that the positive label (in
our case it’s “spam”) has the numeric value of +1 (one), and the negative label (“not_spam”)
has the value of ≠1 (minus one).
At this point, you have a dataset and a learning algorithm, so you are ready to apply
the learning algorithm to the dataset to get the model.
SVM sees every feature vector as a point in a high-dimensional space (in our case, space is 20,000-dimensional). The algorithm puts all feature vectors on an imaginary 20,000-
dimensional plot and draws an imaginary 20,000-dimensional line (a hyperplane) that separates
examples with positive labels from examples with negative labels. In machine learning, the
boundary separating the examples of dierent classes is called the decision boundary.
The equation of the hyperplane is given by two parameters, a real-valued vector w of the
same dimensionality as our input feature vector x, and a real number b like this:
wx ≠ b = 0,
where the expression wx means w(1)x(1) + w(2)x(2) + . . . + w(D)x(D), and D is the number
of dimensions of the feature vector x.
(If some equations aren’t clear to you right now, in Chapter 2 we revisit the math and
statistical concepts necessary to understand them. For the moment, try to get an intuition of
what’s happening here. It all becomes more clear after you read the next chapter.)
Now, the predicted label for some input feature vector x is given like this:
y = sign(wx ≠ b),
where sign is a mathematical operator that takes any value as input and returns +1 if the
input is a positive number or ≠1 if the input is a negative number.
The goal of the learning algorithm — SVM in this case — is to leverage the dataset and find
the optimal values wú and bú for parameters w and b. Once the learning algorithm identifies
these optimal values, the model f (x) is then defined as:
f (x) = sign(wúx ≠ bú)
Therefore, to predict whether an email message is spam or not spam using an SVM model,
you have to take a text of the message, convert it into a feature vector, then multiply this
vector by wú, subtract bú and take the sign of the result. This will give us the prediction (+1
means “spam”, ≠1 means “not_spam”).
Now, how does the machine find wú and bú? It solves an optimization problem. Machines
are good at optimizing functions under constraints.
So what are the constraints we want to satisfy here? First of all, we want the model to predict
the labels of our 10,000 examples correctly. Remember that each example i = 1, . . . , 10000 is
given by a pair (xi, yi), where xi is the feature vector of example i and yi is its label that
takes values either ≠1 or +1. So the constraints are naturally:
• wxi ≠ b Ø 1 if yi = +1, and
• wxi ≠ b Æ ≠1 if yi = ≠1 """)
add_knowledge("""We would also prefer that the hyperplane separates positive examples from negative ones with
the largest margin. The margin is the distance between the closest examples of two classes,
as defined by the decision boundary. A large margin contributes to a better generalization,
that is how well the model will classify new examples in the future. To achieve that, we need
to minimize the Euclidean norm of w denoted by ÎwÎ and given by
ÒqD
j=1(w(j))2.
So, the optimization problem that we want the machine to solve looks like this:
Minimize ÎwÎ subject to yi(wxi ≠ b) Ø 1 for i = 1, . . . , N . The expression yi(wxi ≠ b) Ø 1
is just a compact way to write the above two constraints.
The solution of this optimization problem, given by wú and bú, is called the statistical
model, or, simply, the model. The process of building the model is called training.
For two-dimensional feature vectors, the problem and the solution can be visualized as shown
in fig. 1. The blue and orange circles represent, respectively, positive and negative examples,
and the line given by wx ≠ b = 0 is the decision boundary.
Why, by minimizing the norm of w, do we find the highest margin between the two classes?
Geometrically, the equations wx ≠ b = 1 and wx ≠ b = ≠1 define two parallel hyperplanes,
as you see in fig. 1. The distance between these hyperplanes is given by 2
ÎwÎ , so the smaller the norm ÎwÎ, the larger the distance between these two hyperplanes.
That’s how Support Vector Machines work. This particular version of the algorithm builds
the so-called linear model. It’s called linear because the decision boundary is a straight line
(or a plane, or a hyperplane). SVM can also incorporate kernels that can make the decision
boundary arbitrarily non-linear. In some cases, it could be impossible to perfectly separate
the two groups of points because of noise in the data, errors of labeling, or outliers (examples
very dierent from a “typical” example in the dataset). Another version of SVM can also
incorporate a penalty hyperparameter for misclassification of training examples of specific
classes. We study the SVM algorithm in more detail in Chapter 3.
At this point, you should retain the following: any classification learning algorithm that
builds a model implicitly or explicitly creates a decision boundary. The decision boundary
can be straight, or curved, or it can have a complex form, or it can be a superposition of
some geometrical figures. The form of the decision boundary determines the accuracy of
the model (that is the ratio of examples whose labels are predicted correctly). The form of
the decision boundary, the way it is algorithmically or mathematically computed based on
the training data, dierentiates one learning algorithm from another.
In practice, there are two other essential dierentiators of learning algorithms to consider:
speed of model building and prediction processing time. In many practical cases, you would
prefer a learning algorithm that builds a less accurate model fast. Additionally, you might
prefer a less accurate model that is much quicker at making predictions. """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
add_knowledge(""" """)
# Updating vector representations of the knowledge base for retrieval
update_vectors()

