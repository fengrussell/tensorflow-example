# -*- coding: utf-8 -*-


class Student(object):
    # 之前写的代码经常用这种方式来定义属性，其实这种类属性，每个对象都可以访问。
    # 之前以为是对象变量的操作，后来发现这种集合变量的值在new对象后没有释放，所以不要这么使用，而是用对象变量。
    dict = {}
    atrr = "Student"  # 类属性

    def __init__(self, name ,score):
        self.name = name
        self.score = score
        # 对象属性，因为和类属性同名且对象优先级高，所以访问对象的时候会用这个变量值。不过其他对象如果没有对attr赋值，则都是访问的类属性。
        # self.atrr = "Test"

    # 限制对象的属性，不再这个集合的属性是不能定义的。
    __slots__ = ("name", "age", "score", "_grade")


    @property
    def grade(self):
        return self._grade

    @grade.setter
    def grade(self, value):
        if value > 6:
            raise ValueError("grade can not more than six")
        self._grade = value


if __name__ == "__main__":
    s1 = Student("s1", 90)
    print s1.atrr
    s1.age = 10  # 构造函数没有定义属性，也可以这样来赋值，注意必须在__slots__中
    print s1.age

