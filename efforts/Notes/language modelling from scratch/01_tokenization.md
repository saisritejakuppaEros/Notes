Source lectures Link: 
https://cs336.stanford.edu/lectures/?trace=lecture_01


Byte pair encoding is the major one for the token embeddings. Reducing the tokens length so that u can generate as much as information when you infer.

Tokenization is basically converting string to numerics so that the computer can process the information.


Flow of things
1. Intro to Tokenization
2. Tokenization examples
3. Character Tokenizer
4. Byte Tokenizer
5. Word Tokenizer
6. Bpe Tokenizer


# Intro To Tokenization

- Tokenizer: strings ↔ tokens (indices)

- Character-based, byte-based, word-based tokenization are highly suboptimal

- BPE is an effective heuristic that is data-driven

- Tokenization is a separate step, maybe one day do it end-to-end from bytes...

But whatever solution needs to satisfy:

1. Model (e.g., Transformer) should operate on chunks (abstractions) of the sequence (text, video, DNA, etc.)

2. Chunks should be variable (allocate more model capacity to interesting chunks)




# 2. Tokenization Examples

Tokenization is the process of converting raw text into a sequence of numerical tokens that a machine learning model can understand. Different tokenization strategies produce different token sequences depending on how the text is split.

Consider the following example:

```text
Input:
Hello 😊
```

| Tokenizer | Output |
|-----------|--------|
| Character Tokenizer | `['H', 'e', 'l', 'l', 'o', ' ', '😊']` |
| Byte Tokenizer | `[72, 101, 108, 108, 111, 32, 240, 159, 152, 138]` |
| BPE Tokenizer | Depends on the learned merges. Common substrings such as `"Hello"` may become a single token while the emoji is represented by one or more learned byte tokens. |

Different tokenizers offer different trade-offs:

- **Character Tokenizer:** Simple but produces many tokens.
- **Byte Tokenizer:** Supports every Unicode character without requiring a predefined vocabulary.
- **BPE Tokenizer:** Learns frequently occurring byte sequences, reducing the total number of tokens.

---

# 3. Character Tokenizer

A Character Tokenizer represents every character in a string as its corresponding Unicode code point.

## Encoding

During encoding:

1. Read each character in the string.
2. Convert each character to its Unicode integer using `ord()`.
3. Store the resulting integers as the token sequence.

Example:

```python
text = "Cat"

Encoded:
[67, 97, 116]
```

because

```text
'C' -> 67
'a' -> 97
't' -> 116
```

Implementation:

```python
def encode(self, string: str) -> list[int]:
    return list(map(ord, string))
```

---

## Decoding

To reconstruct the original string:

1. Read every integer token.
2. Convert each Unicode value back into a character using `chr()`.
3. Join all characters together.

Example:

```python
[67, 97, 116]
```

becomes

```text
"Cat"
```

Implementation:

```python
def decode(self, indices: list[int]) -> str:
    return "".join(map(chr, indices))
```

---
## Advantages

- Extremely simple to implement.
- Direct mapping between characters and tokens.
- Supports every Unicode character.
- No training or vocabulary construction is required.

---

## Limitations

- Produces a large number of tokens for long texts.
- Does not capture common words or subwords.
- Higher memory and computational cost during model training.
- Less efficient than modern subword tokenizers such as BPE.

---

# 4. Byte Tokenizer

A Byte Tokenizer represents text as a sequence of **UTF-8 bytes** instead of Unicode characters. Every character is first encoded into its UTF-8 byte representation, and each byte (0–255) becomes a token.

Unlike a Character Tokenizer, where one character corresponds to one token, a Byte Tokenizer may use **multiple tokens for a single character** if that character requires more than one UTF-8 byte (e.g., emojis or characters from non-Latin languages).

---

## Encoding

During encoding:

1. Convert the input string into its UTF-8 byte representation using `encode("utf-8")`.
2. Convert each byte into an integer between **0 and 255**.
3. Store these integers as the token sequence.

Implementation:

```python
def encode(self, string: str) -> list[int]:
    string_bytes = string.encode("utf-8")
    indices = list(map(int, string_bytes))
    return indices
```

---

## Decoding

To reconstruct the original text:

1. Convert the integer tokens back into a byte sequence.
2. Decode the bytes using UTF-8.
3. Return the resulting string.

Implementation:

```python
def decode(self, indices: list[int]) -> str:
    string_bytes = bytes(indices)
    string = string_bytes.decode("utf-8")
    return string
```

---

## Example 1: ASCII Text

Input:

```text
Cat
```

UTF-8 bytes:

```text
'C' -> 67
'a' -> 97
't' -> 116
```

Encoded sequence:

```python
[67, 97, 116]
```

Decoding:

```python
[67, 97, 116]
```

↓

```text
"Cat"
```

---

## Example 2: Unicode Character

Input:

```text
😊
```

UTF-8 encoding:

```text
240 159 152 138
```

Encoded sequence:

```python
[240, 159, 152, 138]
```

Although there is only **one character**, it occupies **four UTF-8 bytes**, resulting in four tokens.

Decoding these bytes reconstructs the original emoji:

```text
😊
```

---

## Example 3: Mixed Text

Input:

```text
Hello 😊
```

Encoded sequence:

```python
[72, 101, 108, 108, 111, 32, 240, 159, 152, 138]
```

Notice that:

- `"Hello "` occupies six bytes.
- `"😊"` occupies four bytes.

---

## Advantages

- Supports every Unicode character without requiring a predefined vocabulary.
- No unknown (OOV) tokens—every text can be represented.
- Simple and deterministic encoding and decoding.
- Forms the foundation for modern tokenizers such as Byte Pair Encoding (BPE).

---

## Limitations

- Produces many tokens for non-ASCII characters.
- Does not capture words or subword patterns.
- Less efficient than learned tokenizers such as BPE.
- Longer token sequences increase computation during model training and inference.

---

## Complexity

| Operation | Time Complexity |
|-----------|-----------------|
| Encoding | O(n) |
| Decoding | O(n) |

where **n** is the number of UTF-8 bytes in the input string.
# 5. Word Tokenizer

A Word Tokenizer represents text as a sequence of **words**, where each word is treated as a single token. The text is first split into words (typically using whitespace or punctuation), and each unique word is mapped to an integer ID using a predefined vocabulary.

Unlike Character and Byte Tokenizers, Word Tokenizers operate at the word level, producing significantly fewer tokens for natural language text.

---

## Encoding

During encoding:

1. Split the input text into individual words.
2. Look up each word in a vocabulary.
3. Replace each word with its corresponding token ID.

For example, suppose the vocabulary is:

```python
{
    "Hello": 0,
    "world": 1,
    "!": 2
}
```

Input:

```text
Hello world !
```

Encoded sequence:

```python
[0, 1, 2]
```

A simple implementation might look like:

```python
def encode(self, string: str) -> list[int]:
    words = string.split()
    return [self.word_to_id[word] for word in words]
```

---

## Decoding

To reconstruct the original text:

1. Convert each token ID back into its corresponding word.
2. Join the words together to form the sentence.

Example:

```python
[0, 1, 2]
```

↓

```text
Hello world !
```

Implementation:

```python
def decode(self, indices: list[int]) -> str:
    words = [self.id_to_word[idx] for idx in indices]
    return " ".join(words)
```

---

## Example

Vocabulary:

```text
"Deep"      -> 0
"Learning" -> 1
"is"        -> 2
"fun"       -> 3
```

Input:

```text
Deep Learning is fun
```

Encoding:

```python
[0, 1, 2, 3]
```

Decoding:

```python
[0, 1, 2, 3]
```

↓

```text
Deep Learning is fun
```

---

## Out-of-Vocabulary (OOV) Problem

One of the biggest limitations of Word Tokenizers is that they can only encode words present in the vocabulary.

Suppose the vocabulary contains:

```text
Hello
world
```

Input:

```text
Hello OpenAI
```

Since `"OpenAI"` is not in the vocabulary, the tokenizer cannot encode it directly. It is usually replaced with a special token such as:

```text
<UNK>
```

Result:

```python
[0, <UNK>]
```

This is known as the **Out-of-Vocabulary (OOV)** problem.

---

## Advantages

- Produces fewer tokens compared to character or byte tokenization.
- Easy to understand and implement.
- Preserves word-level semantics.
- Efficient for tasks with a fixed and limited vocabulary.

---

## Limitations

- Suffers from the Out-of-Vocabulary (OOV) problem.
- Requires storing a potentially very large vocabulary.
- Cannot naturally represent new words, misspellings, or rare words.
- Vocabulary size grows rapidly with language diversity.

---

## Complexity

| Operation | Time Complexity |
|-----------|-----------------|
| Encoding | O(n) |
| Decoding | O(n) |

where **n** is the number of words in the input text.

# 6. Byte Pair Encoding (BPE) Tokenizer

A **Byte Pair Encoding (BPE) Tokenizer** is a **subword tokenizer** that represents text as a sequence of frequently occurring byte sequences rather than individual characters or entire words.

It begins with a vocabulary containing all **256 possible UTF-8 bytes**. During training, it repeatedly finds the most frequent pair of adjacent tokens and merges them into a new token. Over time, common words and subwords become single tokens, reducing the total number of tokens needed to represent text.

This approach combines the advantages of Byte and Word tokenization:

- It can represent **every possible Unicode character** because it starts from bytes.
- It learns **frequent words and subwords**, resulting in shorter token sequences.

---

## How BPE Works

BPE Paper:
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf



The BPE algorithm consists of two phases:

1. **Training**
   - Start with all UTF-8 bytes as the initial vocabulary.
   - Count the frequency of adjacent byte pairs.
   - Merge the most frequent pair into a new token.
   - Repeat this process until the desired vocabulary size is reached.

2. **Tokenization**
   - Convert the input text into UTF-8 bytes.
   - Apply the learned merge rules in order.
   - Replace matching byte pairs with their merged token IDs.

---

## Merge Operation

The core operation in BPE is replacing a pair of adjacent tokens with a newly learned token.

Implementation:

```python
def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    new_indices = []
    i = 0

    while i < len(indices):
        if (
            i + 1 < len(indices)
            and indices[i] == pair[0]
            and indices[i + 1] == pair[1]
        ):
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1

    return new_indices
```

For example,

```text
Original tokens:

[a] [b] [a] [b] [c]

Most frequent pair:

(a, b)

Merge:

(a, b) → X

Result:

[X] [X] [c]
```

---

## Encoding

During encoding:

1. Convert the input string into UTF-8 bytes.
2. Represent each byte as an integer.
3. Apply every learned merge rule sequentially.
4. Return the final token sequence.

Implementation:

```python
def encode(self, string: str) -> list[int]:
    indices = list(map(int, string.encode("utf-8")))

    for pair, new_index in self.params.merges.items():
        indices = merge(indices, pair, new_index)

    return indices
```

---

## Decoding

During decoding:

1. Look up the byte sequence for every token.
2. Concatenate all byte sequences.
3. Decode the resulting bytes using UTF-8.

Implementation:

```python
def decode(self, indices: list[int]) -> str:
    bytes_list = list(map(self.params.vocab.get, indices))
    string = b"".join(bytes_list).decode("utf-8")
    return string
```

---

## Example

Suppose the tokenizer has learned the following merge rules:

```text
'H' + 'e'      → 256
256 + 'l'      → 257
257 + 'l'      → 258
258 + 'o'      → 259
```

Input:

```text
Hello
```

Initial byte sequence:

```python
[72, 101, 108, 108, 111]
```

Applying the merge rules step by step:

```text
[72,101,108,108,111]

↓

[256,108,108,111]

↓

[257,108,111]

↓

[258,111]

↓

[259]
```

The five original byte tokens are compressed into a single learned token.

---

## Compression Ratio

A useful metric for evaluating a tokenizer is the **compression ratio**, defined as the average number of UTF-8 bytes represented by each token.

Implementation:

```python
def get_compression_ratio(string: str, indices: list[int]) -> float:
    num_bytes = len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    return num_bytes / num_tokens
```

The compression ratio is computed as:

$$
\text{Compression Ratio} =
\frac{\text{Number of UTF-8 Bytes}}
{\text{Number of Tokens}}
$$
For example,

Input:

```text
Hello
```

- UTF-8 bytes = **5**
- BPE tokens = **1**

Compression ratio:

```text
5 / 1 = 5.0 bytes per token
```

A higher compression ratio generally indicates that the tokenizer represents text more efficiently.

---

## Advantages

- Supports every Unicode character.
- Learns common words and subwords automatically.
- Produces shorter token sequences than character or byte tokenization.
- Eliminates the Out-of-Vocabulary (OOV) problem.
- Widely used in modern Large Language Models such as GPT.

---

## Limitations

- Requires a training phase to learn merge rules.
- More complex than character or byte tokenization.
- The quality of tokenization depends on the training corpus and vocabulary size.
- Different merge vocabularies produce different tokenizations for the same text.

---

## Complexity

| Operation | Time Complexity |
|-----------|-----------------|
| Training | Depends on corpus size and number of merges |
| Encoding | O(n × m) (naïve implementation), where **n** is the number of bytes and **m** is the number of merge rules |
| Decoding | O(n) |

> **Note:** The encoding implementation shown here is a simple educational implementation that applies merge rules sequentially. Production tokenizers (e.g., OpenAI's `tiktoken`) use optimized data structures and algorithms to perform tokenization much more efficiently.