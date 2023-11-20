import torch
import torch.nn as nn
import string
import random
import time
import math
import matplotlib.pyplot as plt
import unicodedata

# 英文字母和特殊符号
all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker


def readLines(filename):
    with open(filename, 'r') as file:
        lines = file.read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# Unicode -> ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# 构建名字列表
male_names = readLines('data/male.txt')
female_names = readLines('data/female.txt')
all_names = male_names + female_names
n_names = len(all_names)


def inputTensor(line, single=False):
    if single:
        tensor = torch.zeros(1, n_letters)
        tensor[0][all_letters.find(line)] = 1
    else:
        tensor = torch.zeros(len(line), 1, n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][all_letters.find(letter)] = 1
    return tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden


    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


# Loss Function
criterion = nn.NLLLoss()

# Optimizer
learning_rate = 0.005
RNN = RNN(n_letters, 128, n_letters)
# RNN.load_state_dict(torch.load('model.pth'))  # （可选）加载预训练模型
optimizer = torch.optim.SGD(RNN.parameters(), lr=learning_rate)

# 生成名字的最大长度
max_length = 32


def train(inp, target):
    hidden = RNN.initHidden()
    RNN.zero_grad()
    loss = 0
    for i in range(inp.size(0)):
        output, hidden = RNN(inp[i], hidden)
        l = criterion(output, target[i].unsqueeze(0))
        loss += l
    loss.backward()
    optimizer.step()
    return output, loss.item() / inp.size(0)


def generate_one(prefix='A'):
    with torch.no_grad():
        prefix_input = inputTensor(prefix)
        hidden = RNN.initHidden()

        for i in range(len(prefix) - 1):
            _, hidden = RNN(prefix_input[i], hidden)
        input = prefix_input[-1]

        output_name = prefix

        for i in range(max_length):
            output, hidden = RNN(input, hidden)
            topv, topi = output.topk(5)  # 获取前5个最可能的候选字母
            topi = topi[0]  # 选择最可能的字母
            print('\n>>> Top 5 predictions:', end=' ')
            for i in range(5):
                if topi[i] == n_letters - 1:
                    continue
                else:
                    print(all_letters[topi[i]], end=' ')

            if topi[0] == n_letters - 1:
                break
            else:
                letter = all_letters[topi[0]]
                output_name += letter
                input = inputTensor(letter, single=True)

        return output_name


# 将名字转化为目标张量
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


# 随机选择一个训练样本
def randomTrainingExample():
    name = all_names[random.randint(0, n_names - 1)]
    inp = inputTensor(name)
    target = targetTensor(name)
    return inp, target


# 计算时间
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def main():
    n_iters = 10000
    print_every = 500
    plot_every = 100
    all_losses = []
    total_loss = 0

    start = time.time()

    for iter in range(1, n_iters + 1):
        output, loss = train(*randomTrainingExample())
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))
            print(">>> Please input the prefix of the name:")
            prefix = input()
            print("\n>>> The name I generated was: ", generate_one(prefix))
            print("===================================================")

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    # Show the loss over time
    plt.figure()
    plt.plot(all_losses)
    torch.save(RNN.state_dict(), 'naming_rnn.pth')

if __name__ == "__main__":
    main()