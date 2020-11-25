import matplotlib.pyplot as plt

def plot_test_result(test):
	sorted_x = sorted(test, key=lambda kv: kv[0]) 
	x = list(range(50))
	y = list(k[1] for k in sorted_x)
	print(sorted_x)
	sorted_k = sorted(test, key=lambda kv: kv[1])
	print(sorted_k)
	plt.plot(x, y)
	plt.xlabel('epoch num')
	plt.ylabel('domain mAP')
	plt.title('test_result')
	plt.show()

test = [(0, 0.49189779300424646), (1, 0.5138904387583882), (2, 0.536334691286777), (4, 0.5645421289016331), (3, 0.5658359516156337), (5, 0.5707727279143228), (6, 0.5729809018775144), (8, 0.5871151999965678), (7, 0.5907337688146128), (9, 0.5968492444634648), (10, 0.6080814978271548), (11, 0.6241844130311393), (12, 0.6284854817365658), (13, 0.6302649758341037), (16, 0.6397864472691053), (15, 0.6399975535300045), (18, 0.6405742179835137), (17, 0.6486698072446467), (14, 0.6534752561987729), (22, 0.6661780391057547), (42, 0.6673500216433375), (33, 0.6674027973738966), (23, 0.6674991527421578), (48, 0.6685768313347468), (19, 0.6690256113493928), (31, 0.6701044096424472), (47, 0.6702893382117315), (21, 0.6721327301262641), (34, 0.6722607267611632), (39, 0.6731835836014719), (28, 0.6732898732535876), (29, 0.6738632378669075), (27, 0.6738644463650251), (26, 0.6747462409127245), (45, 0.675343059268252), (46, 0.6754493009323576), (35, 0.6762345281109167), (20, 0.6762805875922873), (30, 0.6767393006679027), (44, 0.6768907584798227), (37, 0.677008915501073), (24, 0.6778242119082827), (49, 0.6787143534135133), (38, 0.6791027570047017), (36, 0.6795088227915387), (40, 0.6800957925286214), (25, 0.6817012226743485), (32, 0.6823109049155595), (43, 0.6844323451182417), (41, 0.6874981196941213)]
plot_test_result(test)