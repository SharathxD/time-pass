
## Python

**Key Concepts:**

- **Syntax & Structure:**  
  • Indentation as a block delimiter  
  • Dynamic typing and interpreted execution  

- **Data Types & Variables:**  
  • Numbers (integers, floats)  
  • Strings, Booleans  
  • Collections: lists, tuples, dictionaries, and sets  

- **Control Structures:**  
  • Conditional statements: `if`, `elif`, `else`  
  • Loops: `for` and `while` loops  

- **Functions & Modules:**  
  • Defining functions with `def`  
  • Lambda functions and recursion  
  • Importing modules and using packages  

- **Object-Oriented Programming (OOP):**  
  • Classes and objects  
  • Inheritance, encapsulation, polymorphism, and abstraction  

- **Exception Handling:**  
  • `try`, `except`, `finally`, and custom exception creation  

- **File I/O:**  
  • Reading from and writing to files  

- **Advanced Topics (Basics):**  
  • List comprehensions, generator expressions, and decorators  

**Example Code (Python OOP & Exception Handling):**

```python
# File: example.py
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} makes a sound."

class Dog(Animal):
    def speak(self):
        return f"{self.name} barks."

def main():
    try:
        dog = Dog("Buddy")
        print(dog.speak())
        # Example of file I/O
        with open("sample.txt", "w") as f:
            f.write("Hello, Python!")
    except IOError as e:
        print(f"File error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Execution completed.")

if __name__ == "__main__":
    main()
```

---

## Java

**Key Concepts:**

- **Syntax & Structure:**  
  • Class-based, strongly typed, and compiled language  
  • `public static void main(String[] args)` as the entry point  

- **Data Types & Variables:**  
  • Primitive types (int, float, boolean, etc.)  
  • Object types and wrapper classes  

- **Control Structures:**  
  • Conditional (`if-else`, `switch`)  
  • Loops (`for`, `while`, `do-while`)  

- **Object-Oriented Programming (OOP):**  
  • Classes, objects, constructors  
  • Inheritance, polymorphism, encapsulation, and abstraction  

- **Exception Handling:**  
  • `try-catch-finally` blocks  
  • Throwing and declaring exceptions  

- **Collections Framework & Generics:**  
  • Lists, Sets, Maps, and their iterators  
  • Usage of generics for type safety  

- **Multithreading (Basics):**  
  • Creating threads via `Thread` or `Runnable`  
  • Synchronization basics  

- **File I/O:**  
  • Java I/O streams and NIO package  

**Example Code (Java OOP & Exception Handling):**

```java
// File: Example.java
public class Example {
    public static void main(String[] args) {
        try {
            Dog dog = new Dog("Buddy");
            System.out.println(dog.speak());
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        } finally {
            System.out.println("Execution completed.");
        }
    }
}

class Animal {
    String name;

    Animal(String name) {
        this.name = name;
    }

    public String speak() {
        return name + " makes a sound.";
    }
}

class Dog extends Animal {
    Dog(String name) {
        super(name);
    }

    @Override
    public String speak() {
        return name + " barks.";
    }
}
```

---

## C

**Key Concepts:**

- **Syntax & Structure:**  
  • Use of `#include` for standard libraries  
  • `main()` function as entry point  

- **Data Types & Variables:**  
  • Primitive types (int, float, char, etc.)  
  • Arrays and strings  
  • Structures (`struct`)  

- **Control Structures:**  
  • Conditional (`if`, `else if`, `else`, `switch`)  
  • Loops (`for`, `while`, `do-while`)  

- **Functions & Pointers:**  
  • Function declarations and definitions  
  • Pointer basics, pointer arithmetic, and passing by reference  
  • Dynamic memory allocation with `malloc()` and `free()`  

- **Preprocessor Directives:**  
  • Macros with `#define`  
  • Conditional compilation (`#ifdef`, `#ifndef`)  

- **File I/O:**  
  • Standard functions: `fopen()`, `fprintf()`, `fscanf()`, and `fclose()`  

**Example Code (C with Pointers & File I/O):**

```c
// File: example.c
#include <stdio.h>
#include <stdlib.h>

void printArray(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int size = 5;
    int *array = (int *)malloc(size * sizeof(int));
    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    // Initialize and print array
    for (int i = 0; i < size; i++) {
        array[i] = i * 10;
    }
    printArray(array, size);

    // File I/O example
    FILE *fp = fopen("output.txt", "w");
    if (fp != NULL) {
        fprintf(fp, "Array values: ");
        for (int i = 0; i < size; i++) {
            fprintf(fp, "%d ", array[i]);
        }
        fclose(fp);
    } else {
        fprintf(stderr, "File could not be opened.\n");
    }
    
    free(array);
    return 0;
}
```

---

## MongoDB

**Key Concepts:**

- **NoSQL and Document Orientation:**  
  • Data stored in BSON (binary JSON) documents  
  • Schema-less design enabling flexible data structures  

- **Core Structures:**  
  • **Documents:** Individual records (similar to JSON objects)  
  • **Collections:** Groups of documents  

- **CRUD Operations:**  
  • **Create:** Inserting documents  
  • **Read:** Querying documents using the `find()` method  
  • **Update:** Modifying documents with `updateOne()`, `updateMany()`, etc.  
  • **Delete:** Removing documents with `deleteOne()`, `deleteMany()`  

- **Indexes & Performance:**  
  • Creating indexes to speed up query operations  

- **Aggregation Framework:**  
  • Pipeline for data aggregation and transformation  
  • Stages such as `$match`, `$group`, `$sort`, etc.  

- **Replication & Sharding:**  
  • High availability with replica sets  
  • Horizontal scaling with sharding  

**Example Commands (Mongo Shell):**

```javascript
// Insert a new document into the "users" collection
db.users.insertOne({ name: "Alice", age: 28, city: "New York" });

// Query documents where age is greater than 25
db.users.find({ age: { $gt: 25 } });

// Update a document: change city for user "Alice"
db.users.updateOne({ name: "Alice" }, { $set: { city: "Boston" } });

// Delete a document
db.users.deleteOne({ name: "Alice" });

// Aggregation example: Group users by city and count them
db.users.aggregate([
  { $group: { _id: "$city", count: { $sum: 1 } } }
]);
```

---

## SQL

**Key Concepts:**

- **Database Structure:**  
  • Relational databases with tables, rows, and columns  
  • Primary keys, foreign keys, and unique constraints  

- **Data Types:**  
  • Numeric, character (VARCHAR, TEXT), date/time, boolean, etc.  

- **SQL Commands:**  
  • **Data Definition Language (DDL):**  
    - `CREATE`, `ALTER`, `DROP`  
  • **Data Manipulation Language (DML):**  
    - `SELECT`, `INSERT`, `UPDATE`, `DELETE`  

- **Joins & Relationships:**  
  • INNER JOIN, LEFT/RIGHT OUTER JOIN, FULL OUTER JOIN  
  • Understanding relationships between tables  

- **Aggregate Functions & Grouping:**  
  • Functions like `COUNT()`, `SUM()`, `AVG()`, `MIN()`, `MAX()`  
  • Grouping results with `GROUP BY` and filtering groups with `HAVING`  

- **Subqueries & Nested Queries:**  
  • Using subqueries in SELECT, FROM, or WHERE clauses  

- **Indexes & Views:**  
  • Creating indexes to improve query performance  
  • Using views to simplify complex queries  

**Example Query (SQL with JOIN):**

```sql
-- Create tables for demonstration
CREATE TABLE Departments (
    dept_id INT PRIMARY KEY,
    dept_name VARCHAR(50)
);

CREATE TABLE Employees (
    emp_id INT PRIMARY KEY,
    emp_name VARCHAR(50),
    dept_id INT,
    FOREIGN KEY (dept_id) REFERENCES Departments(dept_id)
);

-- Insert sample data
INSERT INTO Departments (dept_id, dept_name)
VALUES (1, 'Sales'), (2, 'Engineering');

INSERT INTO Employees (emp_id, emp_name, dept_id)
VALUES (101, 'John Doe', 1),
       (102, 'Jane Smith', 2),
       (103, 'Alice Brown', 1);

-- Select query with an INNER JOIN
SELECT e.emp_id, e.emp_name, d.dept_name
FROM Employees e
INNER JOIN Departments d ON e.dept_id = d.dept_id
WHERE e.emp_name LIKE 'J%';
```

---

