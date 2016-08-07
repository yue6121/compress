# -*- coding: utf-8 -*-
"""
Created on Sun May 22 01:10:43 2016

@author: yue
"""

import struct
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

class Node:#定义节点，用于生成树
    def __init__(self):
        self.value = ''
        self.left = None
        self.right = None
        self.frequency = 0
        self.code = ''
        
def give_code(node):#确定节点的编码，左节点编码添加‘0’，右节点添加为‘1’
    if node.left:
        node.left.code = node.code+'0'
        give_code(node.left)
    if node.right:
        node.right.code = node.code+'1'
        give_code(node.right)

def save_code(huffman_map,node):#保存树的信息到字典huffman字典，即保存每个字符的编码
    if not node.left and not node.right:
        huffman_map[node.value] = node.code
    if node.left:
        save_code(huffman_map,node.left)
    if node.right:
        save_code(huffman_map,node.right)

def change_value_to_key(huffmap):#解码是key，value互换
    map = {}
    for (key, value) in huffmap.items():
        map[value] = key
    return map

def encode_huffman(file_input,bit):#huffman编码主函数
    origindata = []#原始数据
    lettermap = {}#字符字典
    for line in open(file_input):
        for i in range(0,len(line),bit):
            if i+bit<len(line):
                char = line[i:i+bit]
            else:
                char = line[i:]
            origindata.append(char)
            if lettermap.get(char):#如果char在字典中
                lettermap[char] += 1#计数加1
            else:
                lettermap[char] = 1#如果不在计数为1
    nodelist = []#node的list，记录lettermap的所有key
    for key in lettermap:
        node = Node()
        node.value = key
        node.frequency = lettermap[key]
        nodelist.append(node)#添加node
    nodelist.sort(cmp=lambda n1, n2: cmp(n1.frequency, n2.frequency))#按频率升序
    #huffman算法的主要思想即建立huffman树
    for i in range(len(nodelist)-1):
        node1 = nodelist[0]
        node2 = nodelist[1]
        node = Node()
        node.left = node1
        node.right = node2
        node.frequency = node1.frequency + node2.frequency
        nodelist[0] = node
        nodelist.pop(1)
        nodelist.sort(cmp=lambda n1, n2: cmp(n1.frequency, n2.frequency))#按频率升序
    root = nodelist[0]
    give_code(root)
    huffman_map = {}
    save_code(huffman_map,root)
    length_map = {}#每个编码长度计数用于可视化
    for key in huffman_map:
        lengh = len(huffman_map[key])
        if length_map.get(lengh):
            length_map[lengh] += lettermap[key]
        else:
            length_map[lengh] = lettermap[key] 
    #可视化
    letters = lettermap.keys()
    letters.sort(reverse=True)
    height = []
    for i in range(len(letters)):
        height.append(lettermap[letters[i]])
        if letters[i]=='\n':
            letters[i]='\\n'
    y_pos = np.arange(len(letters))
    plt.figure(1)
    plt.barh(y_pos, height, height=0.7,align='center',alpha=1)
    plt.yticks(y_pos, letters)
    plt.xlabel('Count')
    plt.ylabel('letter')
    plt.title('Letter count in file')
    plt.savefig('letter_count.jpg')
    #plt.show()
    #Encode length
    lengths = length_map.keys()
    lengths.sort(reverse=True)
    height = []
    for length in lengths:
        height.append(length_map[length])
    y_pos = np.arange(len(lengths))
    plt.figure(2)
    plt.barh(y_pos, height, height=0.7,align='center',alpha=1)
    plt.yticks(y_pos, lengths)
    plt.xlabel('Count')
    plt.ylabel('Encode length')
    plt.title('Encode Length Count')
    plt.savefig('encode_length_count.jpg')
    plt.show()
    #保存到文件
    huffman_map_bytes = pickle.dumps(huffman_map)
    code_data = ''
    for letter in origindata:
        code_data += huffman_map[letter]
    f = open("%s_compress" % file_input,'wb')
    f.write(struct.pack('I', len(huffman_map_bytes)))
    f.write(struct.pack('%ds' % len(huffman_map_bytes), huffman_map_bytes))
    f.write(struct.pack('B', len(code_data) % 8))
    for i in range(0, len(code_data), 8):
        if i + 8 < len(code_data):
            f.write(struct.pack('B', int(code_data[i:i + 8], 2)))
        else:
            # padding
            f.write(struct.pack('B', int(code_data[i:], 2)))
    f.close()
    print "compress finished"

def gen_code(nodelist,start,end):#node编码
    if start==end or start+1==end:
        return
    #确定mid使得左右两部分的差最小
    mid = start#mid表示分开的位置
    preSum = 0
    tailSum = 0
    i= start
    j = end-1
    while(j-i>1):
        if(preSum<tailSum):
            preSum += nodelist[i].frequency
            i += 1
        else:
            tailSum += nodelist[j].frequency
            j -= 1
    if (preSum>tailSum):
        mid = j
    else:
        mid = i
    mid = i+1#mid左边编码为‘0’，右边编码为‘1’
    for i in range(start,mid):
        nodelist[i].code += '0'
    for i in range(mid,end):
        nodelist[i].code +='1'
    gen_code(nodelist,start,mid)
    gen_code(nodelist,mid,end)
    
def encode_fano(file_input,bit):#fano算法的主函数
    origindata = []
    lettermap = {}
    for line in open(file_input):
        for i in range(0,len(line),bit):
            if i+bit<len(line):
                char = line[i:i+bit]
            else:
                char = line[i:]
            origindata.append(char)
            if lettermap.get(char):
                lettermap[char] += 1
            else:
                lettermap[char] = 1
    nodelist = []
    for key in lettermap:
        node = Node()
        node.value = key
        node.frequency = lettermap[key]
        nodelist.append(node)
    nodelist.sort(cmp=lambda n1, n2: cmp(n2.frequency, n1.frequency))#按频率降序
    gen_code(nodelist,0,len(nodelist))
    fano_map = {}#fano编码的字符字典
    for node in nodelist:
        fano_map[node.value] = node.code
    length_map = {}#长度字典
    for key in fano_map:
        lengh = len(fano_map[key])
        if length_map.get(lengh):
            length_map[lengh] += lettermap[key]
        else:
            length_map[lengh] = lettermap[key]
    #可视化
    letters = lettermap.keys()
    letters.sort(reverse=True)
    height = []
    for i in range(len(letters)):
        height.append(lettermap[letters[i]])
        if letters[i]=='\n':
            letters[i]='\\n'
    y_pos = np.arange(len(letters))
    plt.figure(1)
    plt.barh(y_pos, height, height=0.7,align='center',alpha=1)
    plt.yticks(y_pos, letters)
    plt.xlabel('Count')
    plt.ylabel('letter')
    plt.title('Letter count in file')
    plt.savefig('letter_count.jpg')
    #Encode length
    lengths = length_map.keys()
    lengths.sort(reverse=True)
    height = []
    for length in lengths:
        height.append(length_map[length])
    y_pos = np.arange(len(lengths))
    plt.figure(2)
    plt.barh(y_pos, height, height=0.7,align='center',alpha=1)
    plt.yticks(y_pos, lengths)
    plt.xlabel('Count')
    plt.ylabel('Encode length')
    plt.title('Encode Length Count')
    plt.savefig('encode_length_count.jpg')
    plt.show()
    plt.close('all')
    fano_map_bytes = pickle.dumps(fano_map)
    code_data = ''
    for letter in origindata:
        code_data += fano_map[letter]
    f = open("%s_compress" % file_input,'wb')
    f.write(struct.pack('I', len(fano_map_bytes)))
    f.write(struct.pack('%ds' % len(fano_map_bytes), fano_map_bytes))
    f.write(struct.pack('B', len(code_data) % 8))
    for i in range(0, len(code_data), 8):
        if i + 8 < len(code_data):
            f.write(struct.pack('B', int(code_data[i:i + 8], 2)))
        else:
            # padding
            f.write(struct.pack('B', int(code_data[i:], 2)))
    f.close()
    print "compress finished"

def decode(file_input):#解码
    f = open(file_input, 'rb')
    size = struct.unpack('I', f.read(4))[0]
    huffman_map = pickle.loads(f.read(size))
    left = struct.unpack('B', f.read(1))[0]
    data = f.read(1)
    datalist = []    
    while not data == '':
        bdata = bin(struct.unpack('B', data)[0])[2:]
        datalist.append(bdata)
        data = f.read(1)
    f.close()
    for i in range(len(datalist) - 1):
        datalist[i] = '%s%s' % ('0' * (8 - len(datalist[i])), datalist[i])
    datalist[-1] = '%s%s' % ('0' * (left - len(datalist[-1])), datalist[-1])
    encode_data = ''.join(datalist)
    current_code = ''
    huffman_map = change_value_to_key(huffman_map)
    f = open('%s_origin' % file_input, 'w')
    for letter in encode_data:
        current_code += letter
        if huffman_map.get(current_code):
            f.write(huffman_map[current_code])
            current_code = ''
    f.close()
    print 'finished decompressing'
    
if __name__=="__main__":
    c_type = int(raw_input(u'请输入数字，0表示压缩，1表示解压缩：'))
    input_file = raw_input(u'请输入文件名：')
    if c_type==0:
        algorithm = int(raw_input(u'请输入数字选择压缩算法，0表示huffman，1表示fano：'))
        bit = int(raw_input(u'请输入数字压缩字节数：'))
        if algorithm==0:
            encode_huffman(input_file,bit)
            print u'压缩前大小:',os.path.getsize(input_file)
            print u'压缩前大小:',os.path.getsize(input_file+'_compress')
        else:
            encode_fano(input_file,bit)
    else:
        decode(input_file)
    