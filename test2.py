def longestMonotonic(a: list):
    increase_array = [0]
    decrease_array = [0]
    len_array = len(a)
    for i in range(1, len_array):
        if a[i] >= a[i - 1]:
            increase_array.append(increase_array[i - 1] + 1)
        else:
            increase_array.append(0)
        if a[len_array - i - 1] >= a[len_array - i]:
            decrease_array.append(decrease_array[i - 1] + 1)
        else:
            decrease_array.append(0)
    if int(max(increase_array)) > int(max(decrease_array)):
        for i, item in enumerate(increase_array):
            if item == max(increase_array):
                return a[i - item : i + 1]
    else:
        for i, item in enumerate(decrease_array):
            if item == max(decrease_array):
                return a[i - item : i + 1]


a = [1, 3, 3, 7, 4, 7, 8, 2, 3]
print(longestMonotonic(a))
