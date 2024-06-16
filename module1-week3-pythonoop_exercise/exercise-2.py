class Student():
    def __init__(self, name, yob, grade):
        self.__name = name
        self.__yob = yob
        self.__grade = grade

    @property
    def name(self):
        return self.__name

    @property
    def yob(self):
        return self.__yob

    @property
    def grade(self):
        return self.__grade

    def describe(self):
        print(
            f"Student - Name: {self.__name} - YoB: {self.__yob} - Grade: {self.__grade}")


class Teacher():
    def __init__(self, name, yob, subject):
        self.__name = name
        self.__yob = yob
        self.__subject = subject

    @property
    def name(self):
        return self.__name

    @property
    def yob(self):
        return self.__yob

    @property
    def subject(self):
        return self.__subject

    def describe(self):
        print(
            f"Teacher - Name: {self.__name} - YoB: {self.__yob} - Subject: {self.__subject}")


class Doctor():
    def __init__(self, name, yob, specialist):
        self.__name = name
        self.__yob = yob
        self.__specialist = specialist

    @property
    def name(self):
        return self.__name

    @property
    def yob(self):
        return self.__yob

    @property
    def specialist(self):
        return self.__specialist

    def describe(self):
        print(
            f"Doctor - Name: {self.__name} - YoB: {self.__yob} - Specialist: {self.__specialist}")


class Ward():
    def __init__(self, name):
        self.name = name
        self.persons = []

    def add_person(self, person):
        self.persons.append(person)

    def describe(self):
        print(f"Ward Name: {self.name}")

        for i in range(len(self.persons)):
            if (isinstance(self.persons[i], Student)):
                print(
                    f"Student - Name: {self.persons[i].name} - YoB: {self.persons[i].yob} - Grade: {self.persons[i].grade}")
            elif (isinstance(self.persons[i], Teacher)):
                print(
                    f"Teacher - Name: {self.persons[i].name} - YoB: {self.persons[i].yob} - Subject: {self.persons[i].subject}")
            else:
                print(
                    f"Doctor - Name: {self.persons[i].name} - YoB: {self.persons[i].yob} - Specialist: {self.persons[i].specialist}")

    def count_doctor(self):
        freque = 0
        for i in range(len(self.persons)):
            if (isinstance(self.persons[i], Doctor)):
                freque += 1
        return freque

    def sort_age(self):
        self.persons.sort(key=lambda x: x.yob)

    def compute_average(self):
        sum_age = 0
        len_teacher = 0
        for i in range(len(self.persons)):
            if (isinstance(self.persons[i], Teacher)):
                sum_age += self.persons[i].yob
                len_teacher += 1
        return sum_age / len_teacher


if __name__ == "__main__":
    student1 = Student(name="studentA", yob=2010, grade="7")
    student1.describe()

    teacher1 = Teacher(name="teacherA", yob=1969, subject="Math")
    teacher1.describe()

    doctor1 = Doctor(name="doctorA", yob=1945, specialist="Endocrinologists")
    doctor1.describe()

    teacher2 = Teacher(name=" teacherB ", yob=1995, subject=" History ")
    doctor2 = Doctor(name=" doctorB ", yob=1975, specialist=" Cardiologists ")
    ward1 = Ward(name=" Ward1 ")
    ward1.add_person(student1)
    ward1.add_person(teacher1)
    ward1.add_person(teacher2)
    ward1.add_person(doctor1)
    ward1.add_person(doctor2)
    ward1.describe()

    print(f"Number of doctors : { ward1.count_doctor ()}")

    ward1.sort_age()
    print("After sorting Age of Ward1 people")
    ward1.describe()

    print(f"Average year of birth ( teachers ): { ward1 . compute_average ()}")
