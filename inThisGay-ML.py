import numpy as np

gayWords = [
    "gay", "hii kɛ hii", "abagaala ebisiyaga", "gai", "gay rehegua", "同性戀", "ગે", "ομοφυλόφιλος",
    "homoseksuelli", "mosodoma", "isitabane", "homo", "समलिङ्गी", "homofil", "همجنسگرا",
    "homoseksuel", "समलैंगिक", "schwul", "mulombwana", "cɛɲɔgɔnɲini", "ގޭ އެވެ", "moc",
    "ເກ", "gejs", "гей", "gay a ni", "ng'ama chwo", "homosexuell", "umusambanyi", "abaryamana bahuje ibitsina",
    "gėjus", "समलिंगी", "takatāpui", "геј", "pelaka", "സ്വവർഗ്ഗാനുരാഗി", "ڬاي", "gay;", "omosesswali",
    "ижил хүйстэн", "bian nin bla nna nga be fa fite nzra nun", "ہم جنس پرست", "gayi", "ဂေး",
    "đồng tính nam", "umulumendo", "সমকামী", "homofiila", "heñvelreizh", "bakla", "homosexuel",
    "समलैङ्गिकः", "sọmọl", "ၵေႇ", "гаи", "bayot", "omoseksyel", "هاوڕەگەزباز", "khaniis", "ngochani",
    "shoga", "gèidh", "homosexuales", "gej", "هم جنس پرست", "සමලිංගික", "مثلي الجنس", "գեյ",
    "গে", "gay sat jaqiwa", "hommi", "masisi", "aerach", "gey", "gay, ɔbarima ne ɔbea nna",
    "lacoo ma maro laco", "salaf xagole", "homoseksual", "ግብረ ሰዶማዊ", "Агеи", "gei", "geja",
    "gayibɔ", "saalqunnamtii saala walfakkaataa raawwatu", "ସମଲିଙ୍ගୀ", "onibaje", "hoyw",
    "ھەمجىنىسلار", "ߜ߭ߊߦߌ", "nwoke nwere mmasị nwoke", "פריילעך", "ゲイ", "გეი", "同性恋",
    "omushaija omushaija", "Gay", "ಸಲಿಂಗಕಾಮಿ", "समलिंगी अशें म्हण्टात", "ខ្ទើយ", "Argaz",
    "ⴰⵔⴳⴰⵣ", "ஓரின சேர்க்கையாளர்", "เกย์", "omoseksuál", "స్వలింగ సంపర్కుడు", "ge", "geý",
    "ŵasepuka", "eşcinsel", "ግብረሰዶማዊ", "ཕོ་མོ་འདྲ་མཉམ།", "همجنسباز", "ਗੇ", "samkyndur",
    "همجنس گرا", "homossexual", "sunnu-nyɔnu-nyɔnu", "wesoły", "gaynaako", "omosessuâl",
    "tagane", "kāne kāne", "ɗan luwaɗi", "게이", "meleg", "समलैंगिक के बा", "homosexuël", "הומו",
    "gay pipul dɛn"
]

vocab = gayWords

def text_to_vector(text, vocab):
    lower_text = text.lower()
    return np.array([1.0 if v.lower() in lower_text else 0.0 for v in vocab])

train_texts = [
    "This is a gay party",
    "We have a normal meeting",
    "The homo community event is tomorrow",
    "This is just a random sentence",
    "homoseksuel people deserve equal rights",
    "This is a generic sentence with no special word",
    "Let’s gather at a gey festival",
    "Ordinary text without that keyword",
    "homofil support group",
    "Another normal day without keywords",
    "gay a ni is happening tonight",
    "A completely unrelated sentence",
    "homosexuell rights are important",
    "No special words here",
    "This event is gay!",
    "Just a normal day",
    "homoseksual context",
    "A sentence with gey term",
    "A completely unrelated sentence"
]

train_labels = [
    1,  # "This is a gay party"
    0,  # "We have a normal meeting"
    1,  # "The homo community event is tomorrow"
    0,  # "This is just a random sentence"
    1,  # "homoseksuel people deserve equal rights"
    0,  # "This is a generic sentence with no special word"
    1,  # "Let’s gather at a gey festival"
    0,  # "Ordinary text without that keyword"
    1,  # "homofil support group"
    0,  # "Another normal day without keywords"
    1,  # "gay a ni is happening tonight"
    0,  # "A completely unrelated sentence"
    1,  # "homosexuell rights are important"
    0,  # "No special words here"
    1,  # "This event is gay!"
    0,  # "Just a normal day"
    1,  # "homoseksual context"
    1,  # "A sentence with gey term"
    0   # "A completely unrelated sentence"
]

assert len(train_texts) == len(train_labels), "train_texts와 train_labels의 길이가 일치하지 않습니다."

X = np.array([text_to_vector(t, vocab) for t in train_texts])
y = np.array(train_labels).reshape(-1, 1)

input_dim = len(vocab)
hidden_dim1 = 32
hidden_dim2 = 16
hidden_dim3 = 8
output_dim = 1
learning_rate = 0.0001
epochs = 45000

W1 = np.random.randn(input_dim, hidden_dim1) * 0.01
b1 = np.zeros((1, hidden_dim1))
W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.01
b2 = np.zeros((1, hidden_dim2))
W3 = np.random.randn(hidden_dim2, hidden_dim3) * 0.01
b3 = np.zeros((1, hidden_dim3))
W4 = np.random.randn(hidden_dim3, output_dim) * 0.01
b4 = np.zeros((1, output_dim))

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)
mW3, vW3 = np.zeros_like(W3), np.zeros_like(W3)
mb3, vb3 = np.zeros_like(b3), np.zeros_like(b3)
mW4, vW4 = np.zeros_like(W4), np.zeros_like(W4)
mb4, vb4 = np.zeros_like(b4), np.zeros_like(b4)

t = 0

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-10
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

for epoch in range(epochs):
    z1 = X.dot(W1) + b1
    a1 = relu(z1)
    z2 = a1.dot(W2) + b2
    a2 = relu(z2)
    z3 = a2.dot(W3) + b3
    a3 = relu(z3)
    z4 = a3.dot(W4) + b4
    a4 = sigmoid(z4)
    
    loss = binary_cross_entropy(y, a4)
    
    dz4 = a4 - y
    dW4 = a3.T.dot(dz4) / len(y)
    db4 = np.sum(dz4, axis=0, keepdims=True) / len(y)
    
    da3 = dz4.dot(W4.T)
    dz3 = da3 * relu_deriv(z3)
    dW3 = a2.T.dot(dz3) / len(y)
    db3 = np.sum(dz3, axis=0, keepdims=True) / len(y)
    
    da2 = dz3.dot(W3.T)
    dz2 = da2 * relu_deriv(z2)
    dW2 = a1.T.dot(dz2) / len(y)
    db2 = np.sum(dz2, axis=0, keepdims=True) / len(y)
    
    da1 = dz2.dot(W2.T)
    dz1 = da1 * relu_deriv(z1)
    dW1 = X.T.dot(dz1) / len(y)
    db1 = np.sum(dz1, axis=0, keepdims=True) / len(y)
    
    t += 1
    
    mW4 = beta1 * mW4 + (1 - beta1) * dW4
    vW4 = beta2 * vW4 + (1 - beta2) * (dW4 ** 2)
    m_hat_W4 = mW4 / (1 - beta1 ** t)
    v_hat_W4 = vW4 / (1 - beta2 ** t)
    W4 -= learning_rate * m_hat_W4 / (np.sqrt(v_hat_W4) + epsilon)
    
    mb4 = beta1 * mb4 + (1 - beta1) * db4
    vb4 = beta2 * vb4 + (1 - beta2) * (db4 ** 2)
    m_hat_b4 = mb4 / (1 - beta1 ** t)
    v_hat_b4 = vb4 / (1 - beta2 ** t)
    b4 -= learning_rate * m_hat_b4 / (np.sqrt(v_hat_b4) + epsilon)
    
    mW3 = beta1 * mW3 + (1 - beta1) * dW3
    vW3 = beta2 * vW3 + (1 - beta2) * (dW3 ** 2)
    m_hat_W3 = mW3 / (1 - beta1 ** t)
    v_hat_W3 = vW3 / (1 - beta2 ** t)
    W3 -= learning_rate * m_hat_W3 / (np.sqrt(v_hat_W3) + epsilon)
    
    mb3 = beta1 * mb3 + (1 - beta1) * db3
    vb3 = beta2 * vb3 + (1 - beta2) * (db3 ** 2)
    m_hat_b3 = mb3 / (1 - beta1 ** t)
    v_hat_b3 = vb3 / (1 - beta2 ** t)
    b3 -= learning_rate * m_hat_b3 / (np.sqrt(v_hat_b3) + epsilon)
    
    mW2 = beta1 * mW2 + (1 - beta1) * dW2
    vW2 = beta2 * vW2 + (1 - beta2) * (dW2 ** 2)
    m_hat_W2 = mW2 / (1 - beta1 ** t)
    v_hat_W2 = vW2 / (1 - beta2 ** t)
    W2 -= learning_rate * m_hat_W2 / (np.sqrt(v_hat_W2) + epsilon)
    
    mb2 = beta1 * mb2 + (1 - beta1) * db2
    vb2 = beta2 * vb2 + (1 - beta2) * (db2 ** 2)
    m_hat_b2 = mb2 / (1 - beta1 ** t)
    v_hat_b2 = vb2 / (1 - beta2 ** t)
    b2 -= learning_rate * m_hat_b2 / (np.sqrt(v_hat_b2) + epsilon)
    
    mW1 = beta1 * mW1 + (1 - beta1) * dW1
    vW1 = beta2 * vW1 + (1 - beta2) * (dW1 ** 2)
    m_hat_W1 = mW1 / (1 - beta1 ** t)
    v_hat_W1 = vW1 / (1 - beta2 ** t)
    W1 -= learning_rate * m_hat_W1 / (np.sqrt(v_hat_W1) + epsilon)
    
    mb1 = beta1 * mb1 + (1 - beta1) * db1
    vb1 = beta2 * vb1 + (1 - beta2) * (db1 ** 2)
    m_hat_b1 = mb1 / (1 - beta1 ** t)
    v_hat_b1 = vb1 / (1 - beta2 ** t)
    b1 -= learning_rate * m_hat_b1 / (np.sqrt(v_hat_b1) + epsilon)
    
    if (epoch+1) % 5000 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

def isThisGay_ML(text):
    x = text_to_vector(text, vocab).reshape(1, -1)
    z1 = x.dot(W1) + b1
    a1 = relu(z1)
    z2 = a1.dot(W2) + b2
    a2 = relu(z2)
    z3 = a2.dot(W3) + b3
    a3 = relu(z3)
    z4 = a3.dot(W4) + b4
    a4 = sigmoid(z4)
    return bool(a4[0,0] >= 0.5)

test_sentences = [
    "This event is gay!",
    "Just a normal day",
    "homoseksual context",
    "A sentence with gey term",
    "A completely unrelated sentence",
    "homofil support group",
    "No keywords here",
    "Gay a ni is happening tonight"
]

for s in test_sentences:
    pred = isThisGay_ML(s)
    print(f"Text: '{s}' -> Predicted Gay?: {pred}")
