# Отчёт: Генератор текста на базе архитектуры Transformer

## 1. Введение

В рамках курса по машинному обучению было выполнено домашнее задание по созданию генератора текста на основе архитектуры Transformer. Данная работа представляет собой практическое применение механизма самовнимания (self-attention) и декодерной части трансформера для задачи генерации текста.

Генерация текста является фундаментальной задачей в области обработки естественного языка (NLP) и лежит в основе многих современных языковых моделей. В этой работе была реализована архитектура Transformer с нуля на фреймворке PyTorch, обученная на текстовом корпусе книг.

## 2. Что нужно сделать

**Основная задача:** Создать генератор текста на базе архитектуры Transformer, используя только декодерную часть модели.

**Конкретные требования:**

1. **Реализовать архитектуру модели `GeneratorTransformer`**, включающую:
   - Позиционное кодирование (Positional Encoding)
   - Multi-Head Self-Attention механизм
   - Feed-Forward сети
   - Decoder blocks с causal mask (для предотвращения заглядывания в будущее)

2. **Токенизация текста:**
   - Создать или использовать BPE (Byte-Pair Encoding) токенизатор
   - Обучить токенизатор на текстовом корпусе

3. **Подготовка данных:**
   - Создать датасет с использованием скользящего окна (SlidingWindowDataset)
   - Обеспечить сдвиг контекста для авторегрессивного обучения

4. **Обучение модели:**
   - Использовать рекомендованные параметры: batch_size=1, max_length=192, learning_rate=1e-4
   - Провести обучение на 2-4 эпохах
   - Отслеживать динамику функции потерь

5. **Авторегрессивная генерация:**
   - Реализовать метод `generate()` с правильным сдвигом контекста
   - Поддержка temperature sampling для управления разнообразием текста

6. **Тестирование:**
   - Создать чат-интерфейс для интерактивного тестирования модели
   - Провести генерацию текста по заданным промптам

**Дополнительное задание:** Реализовать beam search для улучшения качества генерации.

## 3. Гипотеза

Ожидалось, что:

1. **Модель сможет выучить базовые паттерны языка:** Грамматические структуры, синтаксис, стилистику текста на котором обучалась.

2. **Качество генерации будет зависеть от:**
   - Размера модели (количества параметров)
   - Объёма и качества обучающего текста
   - Количества эпох обучения
   - Параметров генерации (temperature, context_len)

3. **При температуре около 0.7-0.8:** Генерация будет достаточно связной, но с элементом креативности.

4. **Функция потерь будет стабильно снижаться** в процессе обучения, что покажет способность модели обучаться.

5. **Causal mask обеспечит правильную авторегрессию:** Модель будет предсказывать следующий токен, основываясь только на предыдущих.

## 4. Ход работы

### 4.1. Создание токенизатора

Был реализован BPE токенизатор с использованием библиотеки `tokenizers`:

```python
def create_bpe_tokenizer(texts, vocab_size=10000):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.normalizer = normalizers.NFD()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"]
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer
```

**Параметры:**
- Размер словаря: 50257 токенов
- Специальные токены: `<unk>`, `<s>` (BOS), `</s>` (EOS), `<pad>`
- Токенизатор сохранён в файл `custom_tokenizer.json`

### 4.2. Подготовка данных

Создан класс `SlidingWindowDataset` для подготовки обучающих данных:

```python
class SlidingWindowDataset(Dataset):
    def __init__(self, tokenizer, text, max_length=192):
        # Разбиение текста на последовательности длиной max_length
        # input_ids: токены [i : i+max_length-1]
        # labels: токены [i+1 : i+max_length] (сдвинуты на 1)
```

**Ключевые особенности:**
- Используется скользящее окно для создания последовательностей
- Labels сдвинуты на 1 токен вправо относительно input (для обучения предсказанию следующего токена)
- Max length: 192 токена

### 4.3. Реализация архитектуры модели

#### 4.3.1. Positional Encoding

Реализовано синусоидальное позиционное кодирование:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        # Создаётся матрица PE используя sin/cos функции
```

#### 4.3.2. Multi-Head Attention

Реализован механизм самовнимания с несколькими головами:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        # d_model разделяется на num_heads голов
        # Q, K, V матрицы для attention механизма
```

#### 4.3.3. Feed-Forward Network

Двухслойная полносвязная сеть с GELU активацией:

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
```

#### 4.3.4. Transformer Decoder Block

Комбинация Self-Attention и Feed-Forward с нормализацией и residual connections:

```python
class TransformerDecoderBlock(nn.Module):
    def forward(self, x, mask=None):
        attn = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))
        ff = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff))
        return x
```

#### 4.3.5. GeneratorTransformer (Полная модель)

```python
class GeneratorTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, max_length=192, dropout=0.1):
        # Параметры модели:
        # d_model = 384
        # num_heads = 6
        # num_layers = 4
```

**Итоговое количество параметров:** Определяется автоматически при инициализации

#### 4.3.6. Causal Mask

Для обеспечения авторегрессии создаётся causal (нижняя треугольная) маска:

```python
def create_causal_mask(self, size):
    mask = torch.tril(torch.ones(size, size, device=device))
    return mask.unsqueeze(0).unsqueeze(0)
```

### 4.4. Обучение модели

**Гиперпараметры:**
- Batch size: 1 (для экономии памяти)
- Max length: 192 токена
- Learning rate: 1e-4
- Weight decay: 0.01
- Количество эпох: 4
- Оптимизатор: AdamW
- Функция потерь: CrossEntropyLoss (с игнорированием pad токена)
- Gradient clipping: max_norm=1.0

**Процесс обучения:**

```python
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        # Forward pass
        logits = model(input_ids)
        
        # Сдвиг logits и labels для выравнивания
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Вычисление потерь и backpropagation
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), 
                        shift_labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

**Результаты обучения:**
- Потери записывались в список `losses` для визуализации
- График потерь сохранён в `training_loss.png`
- Веса модели сохранены в `generator_transformer.pth`

### 4.5. Авторегрессивная генерация

Реализован метод `generate()` с правильным сдвигом контекста:

```python
def generate(self, prompt, context_len=50, temperature=1.0, max_out_tokens=200):
    # Токенизация промпта
    input_ids = torch.tensor([tokenizer.encode(prompt).ids]).to(device)
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_out_tokens):
            # Сдвиг контекста на context_len токенов
            current_input = generated[:, -context_len:]
            
            # Предсказание следующего токена
            outputs = self(current_input)
            next_token_logits = outputs[0, -1, :] / temperature
            next_token = torch.multinomial(
                F.softmax(next_token_logits, dim=-1), 1
            )
            
            # Добавление токена к результату
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Проверка на EOS токен
            if next_token.item() == eos_token_id:
                break
```

**Ключевые параметры:**
- `context_len`: 64 токена (длина контекста при генерации)
- `temperature`: 0.7-0.8 (управление креативностью)
- `max_out_tokens`: 100-200 (максимальное количество генерируемых токенов)

### 4.6. Тестирование

Реализован чат-интерфейс для интерактивного тестирования:

```python
def chat():
    model.eval()
    while True:
        prompt = input("\nВы: ")
        if prompt.lower() in ['quit', 'exit']:
            break
        response = model.generate(prompt, context_len=64, 
                                  temperature=0.8, max_out_tokens=100)
        print(f"Бот: {response}")
```

**Тестовые промпты:**
- "Привет мир"
- "В начале было"

## 5. Результаты

### 5.1. Обученная модель

- **Архитектура:** Decoder-only Transformer
- **Параметры модели:**
  - d_model: 384
  - num_heads: 6
  - num_layers: 4
  - max_length: 192
  - vocab_size: 50257

### 5.2. Динамика обучения

График функции потерь (`training_loss.png`) показывает изменение loss на протяжении 4 эпох обучения. Снижение loss свидетельствует о том, что модель обучается предсказывать следующий токен.

### 5.3. Сохранённые артефакты

- `generator_transformer.pth` — веса обученной модели
- `custom_tokenizer.json` — обученный BPE токенизатор
- `training_loss.png` — график функции потерь

### 5.4. Генерация текста

Модель способна генерировать текст на основе заданного промпта с использованием:
- Авторегрессивного подхода (токен за токеном)
- Temperature sampling для контроля случайности
- Сдвига контекста для эффективной работы с длинными последовательностями

## 6. Выводы

### 6.1. Достигнутые результаты

1. **Успешно реализована архитектура Transformer с нуля:**
   - Все компоненты (Self-Attention, Positional Encoding, Feed-Forward) были написаны вручную
   - Реализован Decoder-only подход, характерный для генеративных моделей

2. **Создан полный pipeline для обучения:**
   - Токенизатор (BPE)
   - Dataset с скользящим окном
   - DataLoader
   - Цикл обучения с gradient clipping

3. **Реализована авторегрессивная генерация:**
   - Правильный сдвиг контекста
   - Temperature sampling
   - Поддержка EOS токена

4. **Создан интерфейс для тестирования:**
   - Чат-бот для интерактивной проверки модели

### 6.2. Особенности реализации

- **Causal mask** обеспечивает правильную авторегрессию, предотвращая "заглядывание в будущее"
- **Сдвиг контекста** при генерации позволяет модели работать с длинными текстами
- **Temperature** контролирует баланс между связностью (низкая температура) и креативностью (высокая температура)

### 6.3. Возможные улучшения

1. **Beam search:** Дополнительное задание по реализации beam search не было выполнено. Это могло бы улучшить качество генерации.

2. **Mixed precision training:** Не использовался `torch.amp.autocast` для оптимизации памяти и скорости, хотя это было рекомендовано в уроке.

3. **Больший объём данных:** Обучающий текст был небольшим. Использование большего корпуса улучшило бы качество генерации.

4. **Fine-tuning предобученной модели:** Можно было бы использовать предобученный токенизатор от Mistral-7b.

5. **Валидация:** Не использовался валидационный набор для контроля переобучения.

### 6.4. Заключение

В результате работы была успешно реализована и обучена модель GeneratorTransformer на базе архитектуры Transformer. Модель демонстрирует способность к авторегрессивной генерации текста, хотя качество генерации ограничено размером обучающего набора и количеством эпох.

Работа позволила глубоко понять внутренние механизмы Transformer: самовнимание, позиционное кодирование, causal mask, авторегрессивную генерацию. Эти знания являются фундаментом для работы с современными языковыми моделями.
