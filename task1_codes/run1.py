import os
if os.fork():
	os.system('python3 task1.py --device cuda:0 --prompt 0 > ../output/task1/result_points.txt')
elif os.fork():
	os.system('python3 task1.py --device cuda:0 --prompt 1 > ../output/task1/result_multi_points.txt')
elif os.fork():
	os.system('python3 task1.py --device cuda:1 --prompt 2 > ../output/task1/result_bbox.txt')
elif os.fork():
	os.system('python3 task1.py --device cuda:1 --prompt 3 > ../output/task1/result_bbox_margin.txt')