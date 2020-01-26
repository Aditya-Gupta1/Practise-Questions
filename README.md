# Practise Questions

## 1. Check whether 2 strings are anagrams or not.
```python
strings = input().split()
if "".join(sorted(strings[0])) == "".join(sorted(strings[1])):
   print('Anagrams')
else:
  print('No Anangrams')
```