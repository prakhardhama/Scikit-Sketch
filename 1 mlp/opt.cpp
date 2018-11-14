#include <iostream>
#include <sstream>
#include "dataEntry.h"
#include "dataReader.h"
#include "neuralNetwork.h"

using namespace std;

int main()
{	
	freopen("opt_RESULT.txt","w",stdout);
	dataReader dr;
	dr.loadDataFile("optdigits.tra",64,1);
	dr.setCreationApproach( STATIC );
	//in hd op
	neuralNetwork nn(64, 32, 10);
	nn.enableLogging("optdigits.tes");
	//lr mom
	nn.setLearningParameters(0.01, 0.8);
	nn.setDesiredAccuracy(100);
	nn.setMaxEpochs(100);
	dataSet* ds;		
	for (int i=0; i<dr.nDataSets(); ++i)
	{
		ds = dr.getDataSet();	
		nn.trainNetwork( ds->trainingSet, ds->generalizationSet, ds->validationSet );
	}	
}
