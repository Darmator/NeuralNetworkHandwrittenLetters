import numpy
import matplotlib.pyplot
import matplotlib
import scipy.ndimage
from PIL import Image,ExifTags
import PIL


numpy.set_printoptions(suppress=True)
def sigmoid(x):
  return 1/(1+numpy.exp(-x))
def test_image(number, model):
    image_path = "photos/"+str(number)+str(model)+".jpg"
    img = Image.open(image_path)
    img = img.resize((28, 28), PIL.Image.ANTIALIAS)
    image_array = numpy.array(img)
    image_array = image_array[:, :, 0]
    image_array = 255 - image_array
    image_array[image_array<120] = 0
    all_array = image_array.flatten()
    answer = myNeuralNetwork.query((all_array/255.0 * 0.99)+0.01)
    answer_index = list(answer).index(max(answer))
    print("I recognize a "+chr(answer_index+65)+"\n")
    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()
    return answer

def test_file(number_testing):
    test_data_file = open("D:\programacion\mnist_dataset\A_Z Handwritten Data.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    test_all_values = test_data_list[number_testing].split(',')
    print(test_all_values[0])
    image_array = numpy.asfarray(test_all_values[1:]).reshape((28,28))
    answer = myNeuralNetwork.query((numpy.asfarray(test_all_values[1:]) /255.0 * 0.99)+0.01)
    for counter in range(myNeuralNetwork.oNodes):
        probability = str(answer[counter]*100) + "%"
        print(chr(counter+97)+": "+probability)
        pass
    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()
    return answer
def file_test_10000():
    test_data_file = open("D:\programacion\mnist_dataset\A_Z Handwritten Data.csv", 'r')
    
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    amount_questions = len(test_data_list)
    batch_percentage = 1
    test_set = 0
    right_answers = 0
    numpy.random.shuffle(test_data_list)
    for record in test_data_list:
        if(test_set == (len(test_data_list)/10)*batch_percentage ):
            print(str(10*batch_percentage)+ "% DONE")
            batch_percentage+=1
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 *0.99)
        correct_answer = int(all_values[0])
        answer = myNeuralNetwork.query(inputs)
        answer_list = numpy.array(answer) 
        answer_index = list(answer_list).index(max(answer))
        if (answer_index == correct_answer):
            right_answers+=1
        #if (answer_index != correct_answer):
         #   print(str(test_set)+","+str(correct_answer)+","+str(answer_index))
        test_set+=1
        pass
    correct_ratio = (right_answers / amount_questions) * 100
    print("The neural network has a "+str(correct_ratio)+ "%"+" of accuracy")
def training():
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    epochs = 2
    test_set = 0
    numpy.random.shuffle(training_data_list)
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 *0.99)
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            myNeuralNetwork.train(inputs,targets)
            inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01,order=1,reshape=False)
            myNeuralNetwork.train(inputs_plusx_img.reshape(784),targets)
            inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01,order=1,reshape=False)
            myNeuralNetwork.train(inputs_minusx_img.reshape(784),targets)
            test_set+=1
            pass
        print("half")
        pass

class neuralNetwork:
    #initialise neural network
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes
        
        self.lr = learningRate
        
        self.weightIH = numpy.random.normal(0.0, pow(self.inNodes, -0.5), (self.hNodes, self.inNodes))
        self.weightHO = numpy.random.normal(0.0, pow(self.hNodes, -0.5), (self.oNodes, self.hNodes))
        self.activation_function = lambda x: sigmoid(x)
        self.trainSet = 0
        pass
    #train neural network
    def train(self,inputs_list,targets_list):
        targets = numpy.array(targets_list, ndmin=2).T
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        hidden_inputs = numpy.dot(self.weightIH,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.weightHO,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.weightHO.T, output_errors)
        
        self.weightHO += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.weightIH += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
    #take an output from neural network
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        #print(inputs.shape)
        #print(self.weightIH.shape)
        hidden_inputs = numpy.dot(self.weightIH,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.weightHO,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    def save_neural_network(self):
        numpy.savetxt('weightHO.csv', self.weightHO.reshape(1,myNeuralNetwork.oNodes*myNeuralNetwork.hNodes), delimiter=',')   # X is an array
        numpy.savetxt('weightIH.csv', self.weightIH.reshape(1,myNeuralNetwork.inNodes*myNeuralNetwork.hNodes), delimiter=',')   # X is an array
    def load_neural_network(self):
        weightIH_data_file = open("D:/programacion/Python projects/LetterdetectingNeuralNetwork/weightIH.csv", 'r')
        weightIH_data_list = weightIH_data_file.readlines()
        weightIH_data_file.close()
        weightHO_data_file = open("D:/programacion/Python projects/LetterdetectingNeuralNetwork/weightHO.csv", 'r')
        weightHO_data_list = weightHO_data_file.readlines()
        weightHO_data_file.close()
        self.weightIH = numpy.asarray(weightIH_data_list[0].split(','), dtype='float64')
        self.weightIH = self.weightIH.reshape(self.hNodes,self.inNodes)
        self.weightHO = numpy.asarray(weightHO_data_list[0].split(','), dtype='float64')
        self.weightHO = self.weightHO.reshape(self.oNodes,self.hNodes)
input_nodes = 784
hidden_nodes = 200
output_nodes = 26
learning_rate = 0.01

myNeuralNetwork = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
#training_data_file = open("D:\programacion\mnist_dataset\A_Z Handwritten Data.csv", 'r')
#training_data_list = training_data_file.readlines()
#numpy.random.shuffle(training_data_list)
#all_values = training_data_list[0].split(',')
#all_values = numpy.asfarray(all_values[1:])
#all_values = all_values.reshape(28,28) 
#matplotlib.pyplot.imshow(all_values, cmap='Greys', interpolation='None')
#matplotlib.pyplot.show()
#for counter in range (1000):
    #sCounter = counter
    #all_values = training_data_list[sCounter].split(',')
    #print(all_values[0])
    #pass
#training_data_file.close()
training_type = input("Choose 0: Train 1: Load neural network ")
if(int(training_type) == 0):
    training_data_file = open("D:\programacion\mnist_dataset\A_Z Handwritten Data.csv", 'r')
    training()
if(int(training_type) == 1):
    myNeuralNetwork.load_neural_network()


while (1):
    testing_type = input("Choose 0: single file test 1: image test 2: multiple file test 3: save neural network ")
    if (int(testing_type) == 0):
        file_number = input("Enter file number ")
        test_file(int(file_number))
    if (int(testing_type) == 1):
        image_number = input("Enter number ")
        model_number = input("Enter model ")
        test_image(image_number,model_number)
    if (int(testing_type) == 2):
        file_test_10000()
    if (int(testing_type) == 3):
        myNeuralNetwork.save_neural_network()
    pass
