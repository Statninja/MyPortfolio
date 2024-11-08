const aiBook = {
    "Generative AI": [
        "Generative AI is a branch of artificial intelligence that focuses on creating new data that resembles existing data, like text, images, or even sound. It uses machine learning models, primarily neural networks, to generate data that can be creative or informative."
    ],
    "Prompt Engineering": [
        "Prompt engineering is the process of designing inputs for generative models to produce specific outputs. By carefully crafting prompts, users can guide models to create responses that are relevant and accurate. Effective prompts clarify context, specify details, and may ask questions to elicit structured responses."
    ],
    "Training Models": [
        "Training generative models involves feeding them large datasets and adjusting internal parameters so they can understand patterns in the data. Fine-tuning is an additional step where a pre-trained model is refined on a specific dataset to specialize it for a particular domain or style, like legal text or scientific articles."
    ],
    "RLHF": [
        "RLHF is a technique where a model learns to improve responses by receiving feedback from humans. The model is rewarded for producing high-quality answers, encouraging it to generate better responses over time."
    ],
    "Machine Learning Basics": [
        "Machine learning is a subset of AI where computers learn from data to make predictions or decisions. Rather than being explicitly programmed, they improve over time by identifying patterns in data.",
        "In supervised learning, the model is trained on labeled data, where each example has an input and an expected output. The model learns to map inputs to outputs by minimizing errors in predictions. Common algorithms include linear regression, decision trees, and support vector machines.",
        "Unsupervised learning models work with unlabeled data, trying to discover hidden patterns or groupings within the data. Clustering (e.g., k-means) and dimensionality reduction (e.g., PCA) are popular techniques.",
        "Overfitting happens when a model learns too much from the training data, capturing noise rather than the underlying pattern. Underfitting occurs when the model fails to capture enough detail in the data. Balancing model complexity is crucial for optimal performance.",
        "Common metrics for evaluating models include accuracy, precision, recall, and F1 score. Regression models may use metrics like RMSE (Root Mean Square Error) or MAE (Mean Absolute Error). These metrics help determine a model's accuracy and reliability."
    ],
    "Natural Language Processing": [
        "Natural Language Processing is a field of AI that enables computers to understand, interpret, and respond to human language. NLP powers applications like translation, chatbots, and voice assistants.",
        "Tokenization is the process of breaking text into individual units, like words or subwords. Embeddings convert these tokens into numerical vectors, capturing semantic meaning. Word embeddings (e.g., Word2Vec) allow models to understand relationships between words.",
        "Transformers revolutionized NLP by enabling models to pay attention to different parts of a sentence, helping them capture context better. Attention mechanisms assign weight to words based on their relevance to other words in a sentence, improving comprehension.",
        "Question-answering models use NLP techniques to interpret questions and retrieve relevant information from a dataset. By understanding context, these models provide precise answers based on the question's intent."
    ],
    "Neural Networks and Deep Learning": [
        "Neural networks are computing systems inspired by the human brain. They consist of interconnected nodes (neurons) organized into layers. Networks learn by adjusting the connections (weights) between neurons based on training data.",
        "Types of neural networks include feedforward neural networks, convolutional neural networks (CNNs) for image processing, recurrent neural networks (RNNs) for sequential data, and transformers specialized for NLP.",
        "Backpropagation is the process of adjusting weights in the network by calculating the error in predictions. Gradient descent is an optimization algorithm used to minimize this error, adjusting weights iteratively until the model learns the data patterns.",
        "Activation functions introduce non-linearities into neural networks, enabling them to learn complex patterns. Common activation functions include ReLU, sigmoid, and tanh."
    ],
  
    "Ethics in AI": [
        "AI models can inadvertently learn biases present in training data, leading to unfair or discriminatory outcomes. Ensuring data diversity, reviewing model outputs, and adjusting parameters help create fairer AI systems.",
        "Explainable AI (XAI) focuses on making AI decisions understandable to humans. Models need to be interpretable, especially in high-stakes areas like healthcare or finance, where understanding the 'why' behind a decision is essential.",
        "Transparency in AI requires clear documentation of how models are developed, trained, and used. Accountability ensures that developers and organizations take responsibility for their AI models' impacts."
    ],

"Support Vector Machines": [
        "The norm ||w||, the larger the distance between these two hyperplanes. That’s how Support Vector Machines work.",
        "This particular version of the algorithm builds the so-called linear model. It’s called linear because the decision boundary is a straight line (or a plane, or a hyperplane).",
        "SVM can also incorporate kernels that can make the decision boundary arbitrarily non-linear. In some cases, it could be impossible to perfectly separate the two groups of points because of noise in the data, errors in labeling, or outliers (examples very different from a 'typical' example in the dataset).",
        "Another version of SVM can also incorporate a penalty hyperparameter for misclassification of training examples of specific classes.",
        "At this point, you should retain the following: any classification learning algorithm that builds a model implicitly or explicitly creates a decision boundary.",
        "The decision boundary can be straight, curved, have a complex form, or be a superposition of some geometrical figures. The form of the decision boundary determines the accuracy of the model (i.e., the ratio of examples whose labels are predicted correctly).",
        "The form of the decision boundary and the way it is algorithmically or mathematically computed based on the training data differentiates one learning algorithm from another.",
        "In practice, there are two other essential differentiators of learning algorithms to consider: speed of model building and prediction processing time.",
        "In many practical cases, you might prefer a learning algorithm that builds a less accurate model fast. Additionally, you might prefer a less accurate model that is much quicker at making predictions."
    ]

};

