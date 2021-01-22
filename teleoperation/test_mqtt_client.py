#!/usr/bin/python
# -*- coding: utf-8 -*

"""
MQTT Client Test for communication 
"""
import paho.mqtt.client as mqtt
import json
import time


'''
construct the MQTT client for communicating with MQTT broker and realizing data transmission
'''
class MQTTClient:
    def __init__(self, client_id="MQTT_Client", broker_host="192.168.6.131", broker_port=1883, broker_user=None, broker_pass=None, client_keepalive=60):
        # define mqtt parameters
        self.client_id = client_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.broker_user = broker_user
        self.broker_pass = broker_pass
        self.client_keepalive = client_keepalive

        # redefine the topic parameters while dealing with topic
        self.topic_sub = "msg_from_server"
        self.qos_sub = 0
        self.msg_sub = None
        self.topic_pub = "msg_to_server"
        self.qos_pub = 0
        self.msg_pub = None

        # overwrite mqtt functions for specified usage
        self.client = mqtt.Client(self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_subscribe = self.on_subscribe
        self.client.on_unsubscribe = self.on_unsubscribe
        self.client.on_publish = self.on_publish
        self.client.on_message = self.on_message
        self.client.on_log = self.on_log

        # change connection method and loop method
        self.connect_async = self.client.connect_async #asynchronous connect the broker in a non-blocking manner, and the the connection will not complete until loop_start() is called
        self.connect_srv = self.client.connect_srv #connect a broker using an SRV DNS lookup to obtain the broker address
        self.loop = self.client.loop
        self.loop_start = self.client.loop_start
        self.loop_stop = self.client.loop_stop
        self.loop_forever = self.client.loop_forever #blocking network and will not return until the client calls disconnect()


    def connect(self):
        self.client.connect(self.broker_host, self.broker_port, self.client_keepalive)
        print("Connecting to server[%s]......"%self.broker_host)


    def on_connect(self, client, userdata, flags, rc):
        print("Connected to sever[%s] with result code "%self.broker_host + str(rc))


    def disconnect(self):
        self.client.disconnect()
        print("Disconnecting from server[%s]......"%self.broker_host)
    

    def on_disconnect(self, client, userdata, rc):
        print("Disconnected from server[%s] with result code "%self.broker_host + str(rc))
    

    # @staticmethod
    def subscribe(self):
        self.client.subscribe(self.topic_sub, self.qos_sub)
        print("Subscribing topic[%s] from server[%s]......"%(self.topic_sub, self.broker_host))


    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("Subscribed topic[%s] form server[%s]: "%(self.topic_sub, self.broker_host) + str(mid) + " " + str(granted_qos))


    def unsubscribe(self):
        self.client.unsubscribe(self.topic_sub)
        print("Unsubscribing topic[%s] from server[%s]......"%(self.topic_sub, self.broker_host))
    

    def on_unsubscribe(self, client, userdata, mid):
        print("Unsubscribed topic[%s] from server[%s]: "%(self.topic_sub, self.broker_host) + str(mid) + " " + str(self.qos_sub))
    
    # @staticmethod
    def publish(self):
        self.client.publish(self.topic_pub, payload=self.msg_pub, qos=self.qos_pub, retain=False)
        print("Publishing topic[%s] to server[%s]......"%(self.topic_pub, self.broker_host))


    def on_publish(self, client, userdata, mid):
        print("Published topic[%s] to server[%s]: "%(self.topic_pub, self.broker_host) + str(mid))


    def on_message(self, client, userdata, msg):
        try:
            self.msg_sub=json.loads(msg.payload)
            print("Received msg[%s] from server[%s]"%(self.msg_sub, self.broker_host))
        except json.JSONDecodeError as e:
            print("Parsing Msg Error: %s"%e.msg)
    

    def on_log(self, client, userdata, level, string):
        print(string)
        
    
"""
Construct the instance for the MQTT Client and communicate with the broker server for data transmission
"""
# instance the mqtt client and connect to the broker server
mqtt_client = MQTTClient("Jade-PC")
mqtt_client.connect()

# redefine mqtt client topic_pub/topic_sub parameters
mqtt_client.topic_pub = "hand_pose"
mqtt_client.qos_pub = 0
mqtt_client.topic_sub = "ue4_states"
mqtt_client.qos_sub = 0

# loop the subscribe/publish for data transmission
## receive scene data from UE4
mqtt_client.subscribe()
while True:
    mqtt_client.loop()
    ue4_states = mqtt_client.msg_sub #for other usage
    pass
## send hand pose data to UE4
mqtt_client.loop_start()
while True:
    finger_data = 0.0
    hand_pose = get_hand_joint_pose(finger_data)
    mqtt_client.msg_pub = json.dumps(hand_pose)
    mqtt_client.publish()
    pass

