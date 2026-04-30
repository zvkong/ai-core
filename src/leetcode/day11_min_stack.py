class MinStack:

    def __init__(self):
        self.stack = []

    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
        else:
            prev_min = self.stack[-1][1]
            current_min = min(prev_min, val)
            self.stack.append((val, current_min))
    def pop(self) -> None:
        return self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
        


# Your MinStack object will be instantiated and called as such:
obj = MinStack()
val = 2
obj.push(val)
obj.pop()
param_3 = obj.top()
param_4 = obj.getMin()
param_3
param_4

l = [1,2,3,4,5]
min(l)