class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        brackets = { ')':'(', '}':'{', ']':'['}
        for char in s:
            if char in ['(', '[', '{']:
                stack.append(char)
            
            elif char in brackets:
                if len(stack) == 0:
                    return False
            
                top = stack.pop()
                if top != brackets[char]:
                    return False
            else:
                continue

        return len(stack) == 0