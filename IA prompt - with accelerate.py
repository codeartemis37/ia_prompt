# Importations
from collections import Counter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import random
import os
from accelerate import Accelerator
import entrainement_list

# Variables globales
nlp = spacy.load("fr_core_news_sm")
char_to_token = [
    ["'", "<APOS>"],
    [".", "<POINT>"],
    [" ", "<SPACE>"]
]

# Fonctions
def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def modify_phrase(phrase):
    return ''.join(token if char not in phrase else phrase.replace(char, token) for char, token in char_to_token) + '<SPACE><POINT><END>'

def preprocess_data():
    print("Application de la lemmatisation...")
    entrainement = [[lemmatize(element[0]), lemmatize(element[1])] for element in entrainement_list.entrainement]
    print("Prétraitement des données...")
    entrainement = [[modify_phrase(element[0]), modify_phrase(element[1])] for element in entrainement]
    print("Vérification des données d'entraînement...")
    for i, (prompt, response) in enumerate(entrainement):
        if not prompt or not response:
            print(f"Avertissement : Paire invalide à l'index {i}")
            print(f"Prompt: '{prompt}'")
            print(f"Response: '{response}'")
    return entrainement

def create_vocabulary(entrainement):
    print("Création du vocabulaire...")
    vocab = set()
    for prompt, response in entrainement:
        vocab.update(prompt.split())
        vocab.update(response.split())
    vocab = ['<PAD>', '<SPACE>', '<END>', '<POINT>'] + list(vocab)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    return vocab, word_to_ix, ix_to_word

def train_model(model, criterion, optimizer, num_epochs, entrainement, word_to_ix, accelerator):
    print("Début de l'entraînement...")
    model, optimizer, criterion = accelerator.prepare(model, optimizer, criterion)
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)
    for epoch in epoch_pbar:
        total_loss = 0
        batch_pbar = tqdm(entrainement, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False, position=1)
        for prompt, response in batch_pbar:
            prompt_indices = [word_to_ix.get(word, word_to_ix['<PAD>']) for word in prompt.split()]
            response_indices = [word_to_ix.get(word, word_to_ix['<PAD>']) for word in response.split()]
            max_len = max(len(prompt_indices), len(response_indices))
            prompt_tensor = torch.tensor(prompt_indices + [word_to_ix['<PAD>']] * (max_len - len(prompt_indices)), dtype=torch.long).unsqueeze(0)
            response_tensor = torch.tensor(response_indices + [word_to_ix['<PAD>']] * (max_len - len(response_indices)), dtype=torch.long).unsqueeze(0)
            prompt_tensor, response_tensor = accelerator.prepare(prompt_tensor, response_tensor)
            hidden = (torch.zeros(1, 1, model.hidden_dim), torch.zeros(1, 1, model.hidden_dim))
            hidden = accelerator.prepare(hidden)
            optimizer.zero_grad()
            output, _ = model(prompt_tensor, hidden)
            loss = criterion(output.view(-1, model.vocab_size), response_tensor.view(-1))
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
            batch_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        avg_loss = total_loss / len(entrainement)
        epoch_pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})
    print("Entraînement terminé!")

def top_k_sampling(output, k=5):
    top_k = torch.topk(output, k).indices.squeeze(0).tolist()
    return random.choice(top_k)

def generate_response(model, input_text, word_to_ix, ix_to_word, accelerator, max_length=20, k=5):
    model.eval()
    lemmatized_input = lemmatize(modify_phrase(input_text))
    input_indices = [word_to_ix.get(word, word_to_ix['<PAD>']) for word in lemmatized_input.split()]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0)
    hidden = (torch.zeros(1, 1, model.hidden_dim), torch.zeros(1, 1, model.hidden_dim))
    input_tensor, hidden = accelerator.prepare(input_tensor, hidden)
    generated_words = []
    for _ in range(max_length):
        output, hidden = model(input_tensor, hidden)
        word_idx = top_k_sampling(output.squeeze(0)[-1], k=k)
        generated_word = ix_to_word[word_idx]
        if generated_word == '<END>':
            break
        generated_words.append(generated_word)
        input_tensor = torch.tensor([[word_idx]], dtype=torch.long).to(accelerator.device)
    generated_text = "".join(generated_words)
    for char, token in char_to_token:
        generated_text = generated_text.replace(token, char)
    return generated_text

# Classes
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
    
    def forward(self, x, hidden):
        embeds = self.embedding(x)
        embeds = self.dropout(embeds)
        output, hidden = self.lstm(embeds, hidden)
        output = self.fc(output)
        return output, hidden

# Code principal
def main():
    accelerator = Accelerator()
    entrainement = preprocess_data()
    vocab, word_to_ix, ix_to_word = create_vocabulary(entrainement)
    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_dim = 300
    learning_rate = 0.002
    num_epochs = 5
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_ix['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_model(model, criterion, optimizer, num_epochs, entrainement, word_to_ix, accelerator)
    print("AI trained and ready!")
    command_actions = {
        '/q': lambda: 'quit',
        '/quit': lambda: 'quit',
        '/entrainement': lambda: print(entrainement),
        '/input_user': lambda: print(lemmatize(user_input)),
        '/clear': lambda: os.system('cls' if os.name == 'nt' else 'clear'),
        '/credits': lambda: print(''' ╭──────────────────────────────────────╮ │ credits │ ├──────────────────────────────────────┤ │ │ │ AI Robot Asimov │ │ by artemis37 │ ╰──────────────────────────────────────╯ '''),
        '/help': lambda: print(''' /q ou /quit: quitte l'IA /entrainement: afficher la liste d'entrainement parsée /input_user: afficher l'input utilisateur parsé /help: affiche cette aide /credits: affiche cette les credits /clear: clear la console ''')
    }
    while True:
        user_input = input("Vous: ")
        action = command_actions.get(user_input.lower(), lambda: 'continue')
        result = action()
        if result == 'quit':
            break
        elif result == 'continue':
            response = generate_response(model, user_input, word_to_ix, ix_to_word, accelerator)
            print("AI:", response)
    print("Au revoir!")

if __name__ == "__main__":
    main()
