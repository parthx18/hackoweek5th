"""
====================================================
Mini-Project: Student Grade Calculator
Week 3 — Python Essentials & NumPy
====================================================
Concepts used:
  ✅ OOP (classes, inheritance, __repr__)
  ✅ Functions
  ✅ List & dict comprehensions
  ✅ NumPy: arrays, vectorized ops, aggregation
====================================================
Run: python grade_calculator.py
"""

import numpy as np


# ====================================================
# OOP — Student Class
# ====================================================
class Student:
    """Represents a student with multiple subject scores."""

    def __init__(self, name, scores: dict):
        """
        Args:
            name   : Student's name (str)
            scores : Dict of subject -> marks e.g. {'Math': 85, 'Science': 90}
        """
        self.name = name
        self.scores = scores
        # NumPy array of all score values for fast computation
        self._arr = np.array(list(scores.values()), dtype=float)

    # ---- Computed properties ----
    @property
    def average(self):
        return np.mean(self._arr)

    @property
    def highest(self):
        return np.max(self._arr)

    @property
    def lowest(self):
        return np.min(self._arr)

    @property
    def std_dev(self):
        return np.std(self._arr)

    @property
    def grade(self):
        """Letter grade based on average."""
        avg = self.average
        if avg >= 90: return 'A'
        elif avg >= 80: return 'B'
        elif avg >= 70: return 'C'
        elif avg >= 60: return 'D'
        else: return 'F'

    @property
    def passed(self):
        """Student passes if average >= 60."""
        return self.average >= 60

    def subject_report(self):
        """Dict comprehension: subject -> pass/fail."""
        return {
            subject: "✅ Pass" if score >= 60 else "❌ Fail"
            for subject, score in self.scores.items()
        }

    def __repr__(self):
        return f"Student(name={self.name!r}, avg={self.average:.1f}, grade={self.grade!r})"


# ====================================================
# Subclass — ScholarshipStudent (Inheritance)
# ====================================================
class ScholarshipStudent(Student):
    """A student who also has a scholarship amount."""

    def __init__(self, name, scores, scholarship_amount):
        super().__init__(name, scores)
        self.scholarship_amount = scholarship_amount

    def __repr__(self):
        base = super().__repr__()
        return f"{base[:-1]}, scholarship=₹{self.scholarship_amount})"


# ====================================================
# Functions
# ====================================================
def print_separator(char="─", width=55):
    print(char * width)


def print_student_report(student: Student):
    """Print a detailed report card for one student."""
    print_separator()
    tag = "🏆" if isinstance(student, ScholarshipStudent) else "📋"
    print(f"{tag}  {student.name}")
    print_separator()

    # List comprehension: format each subject score
    subject_lines = [
        f"   {subj:<12} {score:>5.1f}  {status}"
        for (subj, score), status in zip(
            student.scores.items(),
            student.subject_report().values()
        )
    ]
    for line in subject_lines:
        print(line)

    print_separator("·")
    print(f"   Average   : {student.average:.2f}")
    print(f"   Highest   : {student.highest:.0f}")
    print(f"   Lowest    : {student.lowest:.0f}")
    print(f"   Std Dev   : {student.std_dev:.2f}")
    print(f"   Grade     : {student.grade}")
    print(f"   Status    : {'✅ PASSED' if student.passed else '❌ FAILED'}")

    if isinstance(student, ScholarshipStudent):
        print(f"   Scholarship: ₹{student.scholarship_amount:,}")
    print()


def class_summary(students: list):
    """
    Compute class-wide statistics using NumPy.
    Uses vectorized operations across all student averages.
    """
    # List comprehension to get all averages as a NumPy array
    all_averages = np.array([s.average for s in students])

    # Grade distribution using dict comprehension
    grade_counts = {
        grade: sum(1 for s in students if s.grade == grade)
        for grade in ['A', 'B', 'C', 'D', 'F']
    }

    # Passing students using list comprehension + filter
    passing = [s.name for s in students if s.passed]
    failing = [s.name for s in students if not s.passed]

    # Top student (vectorized argmax)
    top = students[int(np.argmax(all_averages))]

    print_separator("═")
    print("  📊  CLASS SUMMARY")
    print_separator("═")
    print(f"  Total Students  : {len(students)}")
    print(f"  Class Average   : {np.mean(all_averages):.2f}")
    print(f"  Highest Average : {np.max(all_averages):.2f}  ({top.name})")
    print(f"  Lowest Average  : {np.min(all_averages):.2f}")
    print(f"  Std Deviation   : {np.std(all_averages):.2f}")
    print()
    print("  Grade Distribution:")
    for grade, count in grade_counts.items():
        bar = "█" * count
        print(f"    {grade} : {bar} ({count})")
    print()
    print(f"  ✅ Passing ({len(passing)}): {', '.join(passing) or 'None'}")
    print(f"  ❌ Failing ({len(failing)}): {', '.join(failing) or 'None'}")
    print_separator("═")


# ====================================================
# Main — Run the Grade Calculator
# ====================================================
def main():
    print("\n" + "═" * 55)
    print("  🎓  STUDENT GRADE CALCULATOR")
    print("  Week 3 Mini-Project — Python + NumPy")
    print("═" * 55 + "\n")

    # Create student objects
    students = [
        Student("Alice", {
            "Math": 92, "Science": 88, "English": 95,
            "History": 78, "Computer": 97
        }),
        Student("Bob", {
            "Math": 65, "Science": 72, "English": 58,
            "History": 80, "Computer": 70
        }),
        ScholarshipStudent("Charlie", {
            "Math": 98, "Science": 95, "English": 91,
            "History": 93, "Computer": 99
        }, scholarship_amount=50000),
        Student("Diana", {
            "Math": 45, "Science": 50, "English": 55,
            "History": 48, "Computer": 52
        }),
        Student("Eve", {
            "Math": 80, "Science": 83, "English": 77,
            "History": 85, "Computer": 88
        }),
    ]

    # Print individual reports
    print("  INDIVIDUAL REPORT CARDS\n")
    for student in students:
        print_student_report(student)

    # Print class summary (NumPy stats across all students)
    class_summary(students)

    # Bonus: sorted leaderboard using list comprehension + sorted()
    print("\n  🏅  LEADERBOARD")
    print_separator()
    leaderboard = sorted(students, key=lambda s: s.average, reverse=True)
    for rank, s in enumerate(leaderboard, 1):
        medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else f"  {rank}."
        print(f"  {medal}  {s.name:<12} {s.average:.2f}  Grade: {s.grade}")
    print()


if __name__ == "__main__":
    main()
