def find_max_in_file(filename):
    with open(filename, 'r') as file:
        # 读取文件的每一行，并将它们转换为浮点数（或整数，根据你的需要）
        # 使用列表推导式和max函数找到最大值
        # 注意：这里假设每行只有一个数字，并且文件中没有非数字行
        numbers = [float(line.strip()) for line in file if line.strip().replace('.', '', 1).isdigit()]
        if numbers:  # 确保列表中至少有一个数字
            return max(numbers)
        else:
            return None  # 或返回一个错误消息，表示文件中没有数字

def find_min_in_file(filename):
    with open(filename, 'r') as file:
        # 读取文件的每一行，并将它们转换为浮点数（或整数，根据你的需要）
        # 使用列表推导式和max函数找到最大值
        # 注意：这里假设每行只有一个数字，并且文件中没有非数字行
        numbers = [float(line.strip()) for line in file if line.strip().replace('.', '', 1).isdigit()]
        if numbers:  # 确保列表中至少有一个数字
            return min(numbers)
        else:
            return None  # 或返回一个错误消息，表示文件中没有数字

def find_200_in_file(filename):
    with open(filename, 'r') as file:
        # 读取文件的每一行，并将它们转换为浮点数（或整数，根据你的需要）
        # 使用列表推导式和max函数找到最大值
        # 注意：这里假设每行只有一个数字，并且文件中没有非数字行
        numbers = [float(line.strip()) for line in file if line.strip().replace('.', '', 1).isdigit()]
        if numbers:  # 确保列表中至少有一个数字
            return numbers[4]

# 使用函数
for i in [5, 10, 15, 20, 25, 30]:
    filename = 'logs/cost_lfu_agent_'+str(i) +'_cache_390_req_390.txt'
    max_value = find_200_in_file(filename)
    #min_value = find_min_in_file(filename)
    print(max_value)
    #print(min_value)