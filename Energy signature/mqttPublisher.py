import paho.mqtt.client as PahoMQTT
import time
import pandas as pd
import json

class MyPublisher:
	def __init__(self, clientID):
		self.clientID = clientID
		# create an instance of paho.mqtt.client
		self._paho_mqtt = PahoMQTT.Client(self.clientID, False) 
		# register the callback
		self._paho_mqtt.on_connect = self.myOnConnect
		self.messageBroker = 'localhost'

	def start (self):
		#manage connection to broker
		self._paho_mqtt.connect(self.messageBroker, 1883)
		self._paho_mqtt.loop_start()

	def stop (self):
		self._paho_mqtt.loop_stop()
		self._paho_mqtt.disconnect()

	def myPublish(self, topic, message):
		# publish a message with a certain topic
		self._paho_mqtt.publish(topic, message, 2)

	def myOnConnect (self, paho_mqtt, userdata, flags, rc):
		print ("Connected to %s with result code: %d" % (self.messageBroker, rc))



if __name__ == "__main__":
	test = MyPublisher("MyPublisher")
	test.start()
	df=pd.read_csv('processed.csv',sep=',',decimal=',',index_col=0)
	df.index=pd.to_datetime(df.index,unit='s')
	GATEWAY_NAME="residential"
	for i in df.index:
		for j in df.loc[i].items():
			if 'Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)'in j[0] or 'Mean Operative Temperature' in j[0] or 'Electricity:Facility' in j[0] or 'DistrictCooling:Facility' in j[0] or 'DistrictHeating:Facility' in j[0] or "Total energy" in j[0]:
				if pd.isna(j[1]):
					continue
				else:	 
					nodeID=j[0]
					value=j[1]
					if 'Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)'in nodeID:
						measurement="Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)"
					elif 'Electricity:Facility'in nodeID:
						measurement="Electricity:Facility"
					elif 'Mean Operative Temperature'in nodeID:
						measurement="Mean Operative Temperature"
					elif 'DistrictCooling:Facility'in nodeID:
						measurement="DistrictCooling:Facility"
					elif 'DistrictHeating:Facility'in nodeID:		
						measurement="DistrictHeating:Facility"    
					elif 'Total energy'in nodeID:
						measurement="Total energy"
					else:
						measurement="other"
					payload={
								"location":str(GATEWAY_NAME),
								"measurement":measurement,
								"node":str(nodeID),
								"time_stamp":str(i),
								"value":value}
					test.myPublish ('ict4bd2022', json.dumps(payload)) 	
					time.sleep(0.1)
			else :
				continue
	test.stop()


