from kafka import KafkaConsumer, KafkaProducer
import json

class Consumer(KafkaConsumer):
    def __init__(self, topicName, kafkaPort, consumerGroup):
        KafkaConsumer.__init__(
            self,
            topicName,
            bootstrap_servers=kafkaPort,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id=consumerGroup,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )

class Producer(KafkaProducer):
    def __init__(self, kafkaPort):
        KafkaProducer.__init__(
            self,
            bootstrap_servers=[kafkaPort],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
    
    def send(self, topicName, value):
        KafkaProducer.send(self, topicName, value).add_callback(self.on_send_success) 

    def on_send_success(self, record_metadata):
        print("----- on success -----")
        print("Topic:", record_metadata.topic, ", Partition:", record_metadata.partition, ", Offset:", record_metadata.offset)
        print("----------------------")

# class Consumer(multiprocessing.Process):
#     def __init__(self):
#         multiprocessing.Process.__init__(self)
#         self.stop_event = multiprocessing.Event()
        
#     def stop(self):
#         self.stop_event.set()
        
#     def run(self):
#         consumer = KafkaConsumer(bootstrap_servers='localhost:9092',
#                                  auto_offset_reset='earliest',
#                                  consumer_timeout_ms=1000)
#         consumer.subscribe(['my-topic'])

#         while not self.stop_event.is_set():
#             for message in consumer:
#                 print(message)
#                 if self.stop_event.is_set():
#                     break

#         consumer.close()