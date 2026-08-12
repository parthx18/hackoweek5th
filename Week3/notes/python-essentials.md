# Python Essentials — Notes

## 1. Functions

Functions let you reuse blocks of code.

```python
# Basic function
def greet(name):
    return f"Hello, {name}!"

# Default argument
def greet(name="Stranger"):
    return f"Hello, {name}!"

# Multiple return values
def min_max(numbers):
    return min(numbers), max(numbers)

low, high = min_max([3, 1, 9, 4])
```

### *args and **kwargs
```python
def total(*args):          # Variable number of positional args
    return sum(args)

def profile(**kwargs):     # Variable number of keyword args
    for key, val in kwargs.items():
        print(f"{key}: {val}")

total(1, 2, 3, 4)          # → 10
profile(name="Alice", age=21)
```

### Lambda (anonymous functions)
```python
square = lambda x: x ** 2
square(5)  # → 25

# Common with sorted/filter/map
nums = [3, 1, 4, 1, 5]
sorted_nums = sorted(nums, key=lambda x: -x)  # descending
```

---

## 2. OOP — Object Oriented Programming

### Class Basics
```python
class Student:
    # Class variable (shared by all instances)
    school = "Hackoweek University"

    # Constructor
    def __init__(self, name, grade):
        self.name = name      # instance variable
        self.grade = grade

    # Instance method
    def introduce(self):
        return f"I'm {self.name}, grade: {self.grade}"

    # String representation
    def __repr__(self):
        return f"Student(name={self.name}, grade={self.grade})"


# Create objects
alice = Student("Alice", 88)
bob   = Student("Bob", 75)

print(alice.introduce())   # I'm Alice, grade: 88
print(Student.school)      # Hackoweek University
```

### Inheritance
```python
class TopStudent(Student):
    def __init__(self, name, grade, scholarship):
        super().__init__(name, grade)   # call parent __init__
        self.scholarship = scholarship

    def introduce(self):                # override parent method
        base = super().introduce()
        return f"{base} | Scholarship: {self.scholarship}"

diana = TopStudent("Diana", 95, "Merit")
print(diana.introduce())
```

### Encapsulation (private attributes)
```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance   # __ makes it private

    def deposit(self, amount):
        self.__balance += amount

    def get_balance(self):         # getter
        return self.__balance
```

---

## 3. List Comprehensions

A concise way to create lists.

```python
# Basic syntax: [expression for item in iterable if condition]

# Regular loop
squares = []
for x in range(10):
    squares.append(x ** 2)

# List comprehension (same result, one line)
squares = [x ** 2 for x in range(10)]

# With condition
evens = [x for x in range(20) if x % 2 == 0]

# Nested
matrix = [[i * j for j in range(1, 4)] for i in range(1, 4)]
# [[1,2,3], [2,4,6], [3,6,9]]
```

---

## 4. Dict Comprehensions

```python
# { key: value for item in iterable }

students = ["Alice", "Bob", "Charlie"]
grades   = [88, 75, 92]

# Map name → grade
grade_map = {s: g for s, g in zip(students, grades)}
# {'Alice': 88, 'Bob': 75, 'Charlie': 92}

# Filter only passing students (grade >= 80)
passing = {k: v for k, v in grade_map.items() if v >= 80}
# {'Alice': 88, 'Charlie': 92}
```

---

## 5. Useful Built-ins

```python
# map — apply function to each element
doubled = list(map(lambda x: x * 2, [1, 2, 3]))   # [2, 4, 6]

# filter — keep elements that pass a test
passing = list(filter(lambda x: x >= 80, [88, 60, 75, 92]))

# zip — combine two lists
pairs = list(zip(["a", "b", "c"], [1, 2, 3]))  # [('a',1), ('b',2), ('c',3)]

# enumerate — get index + value
for i, name in enumerate(["Alice", "Bob"]):
    print(i, name)   # 0 Alice, 1 Bob
```
