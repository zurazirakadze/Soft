"""
Decision Tree Classification.
"""
from __future__ import print_function

from pyspark import SparkContext
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

import json
from bson import json_util
from bson.json_util import dumps

if __name__ == "__main__":

	sc = SparkContext(appName="DecisionTreeClassification")

	raw_data = MLUtils.loadLibSVMFile(sc, '/home/hechem/spark-campaign-classification/test/data/sample_libsvm_data.txt')
	(trainingDataSet, testDataSet) = raw_data.randomSplit([0.7, 0.3])

	tree = DecisionTree.trainClassifier(trainingDataSet, numClasses=2, categoricalFeaturesInfo={}, impurity='gini', maxDepth=4, maxBins=30)

	predictions = tree.predict(testDataSet.map(lambda x: x.features))
	labelsAndPredictions = testDataSet.map(lambda lp: lp.label).zip(predictions)
	testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testDataSet.count())
	print('Test Error = ' + str(testErr))
	print('Learned classification tree model:')
	print(tree.toDebugString())
	tree_to_json = tree.toDebugString()
	
	# Parser
	def parse(lines):
		block = []
		while lines :
			
			if lines[0].startswith('If'):
				bl = ' '.join(lines.pop(0).split()[1:]).replace('(', '').replace(')', '')
				block.append({'name':bl, 'children':parse(lines)})
				
				
				if lines[0].startswith('Else'):
					be = ' '.join(lines.pop(0).split()[1:]).replace('(', '').replace(')', '')
					block.append({'name':be, 'children':parse(lines)})
			elif not lines[0].startswith(('If','Else')):
				block2 = lines.pop(0)
				block.append({'name':block2})
			else:
				break	
		return block
	
	# Convert Tree to JSON
	def tree_json(tree):
		data = []
		for line in tree.splitlines() : 
			if line.strip():
				line = line.strip()
				data.append(line)
			else : break
			if not line : break
		res = []
		res.append({'name':'Root', 'children':parse(data[1:])})
		with open('/home/hechem/spark-campaign-classification/test/data/structure.json', 'w') as outfile:
			json.dump(res[0], outfile)
		print ('Conversion Success !')
	tree_json(tree_to_json)