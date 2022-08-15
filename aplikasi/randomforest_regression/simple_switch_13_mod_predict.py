# Copyright (C) 2011 Nippon Telegraph and Telephone Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
# penggabungan dirandom index saat data train n data test
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from operator import attrgetter

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ipv6
from ryu.lib.packet import ipv4
from ryu.lib.packet import tcp
from ryu.lib.packet import udp
from ryu.lib.packet import icmp
from ryu.lib.packet import ether_types
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


filepath2 = "/home/aybe/Desktop/coding+data/dataset/testdataset.csv"

test = pd.read_csv(filepath2)

test = test[test.notnull()]

test.head()

feat_labels = list(test.columns)
print(feat_labels)

print(len(test))

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
spec = preprocessing.LabelEncoder()

# ['total_length', 'flags', 'csum', 'src_ip', 'src_port', 'port_no', 'rx_bytes_ave', 'tx_bytes_ave']

flags = preprocessing.LabelEncoder()
src_ip = preprocessing.LabelEncoder()

X_test = test.drop(columns=['label'])

list_features = [5, 10]
selected_labels = [x for i,x in enumerate(feat_labels) if i in list_features]
print(selected_labels)

newX_test = test[selected_labels]

newX_test = newX_test.apply(le.fit_transform)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

newX_important_test = scaler.fit_transform(newX_test)

# ['total_length', 'flags', 'csum', 'src_ip', 'src_port', 'port_no', 'rx_bytes_ave', 'tx_bytes_ave']

flags.fit(X_test['flags'])
src_ip.fit(X_test['src_ip'])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

newX_important_test = scaler.fit_transform(newX_test)

class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}

        # IPV4
        self.flags = ""
        self.csum = ""
        self.src_ip = ""
        self.i = 0

        # TCP & UDP
        self.src_port = "0"

        self.filename6 = "/home/aybe/Desktop/coding+data/model/svmrbf.sav"
        self.filename7 = "/home/aybe/Desktop/coding+data/model/knn.sav"
        self.filename8 = "/home/aybe/Desktop/coding+data/model/dtc.sav"
        self.filename9 = "/home/aybe/Desktop/coding+data/model/rfc.sav"
        self.filename10 = "/home/aybe/Desktop/coding+data/model/mlp.sav"
        self.filename11 = "/home/aybe/Desktop/coding+data/model/adb.sav"
        self.filename12 = "/home/aybe/Desktop/coding+data/model/gnb.sav"
        self.filename14 = "/home/aybe/Desktop/coding+data/model/svmlin.sav"

        self.svmrbf = pickle.load(open(self.filename6, 'rb'))
        self.knn = pickle.load(open(self.filename7, 'rb'))
        self.dtc = pickle.load(open(self.filename8, 'rb'))
        self.rfc = pickle.load(open(self.filename9, 'rb'))
        self.mlp = pickle.load(open(self.filename10, 'rb'))
        self.adb = pickle.load(open(self.filename11, 'rb'))
        self.gnb = pickle.load(open(self.filename12, 'rb'))
        self.svmlin = pickle.load(open(self.filename14, 'rb'))

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # install table-miss flow entry
        #
        # We specify NO BUFFER to max_len of the output action due to
        # OVS bug. At this moment, if we specify a lesser number, e.g.,
        # 128, OVS will send Packet-In with invalid buffer_id and
        # truncated packet data. In that case, we cannot output packets
        # correctly.  The bug has been fixed in OVS v2.1.0.
        match = parser.OFPMatch()
        actions = [
            parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)
        ]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                buffer_id=buffer_id,
                priority=priority,
                match=match,
                instructions=inst,
            )
        else:
            mod = parser.OFPFlowMod(
                datapath=datapath, priority=priority, match=match, instructions=inst
            )
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        # If you hit this you might want to increase
        # the "miss_send_length" of your switch
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug(
                "packet truncated: only %s of %s bytes",
                ev.msg.msg_len,
                ev.msg.total_len,
            )
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        IPV4 = pkt.get_protocols(ipv4.ipv4)
        IPV6 = pkt.get_protocols(ipv6.ipv6)
        UDP = pkt.get_protocols(udp.udp)
        TCP = pkt.get_protocols(tcp.tcp)
        ICMP = pkt.get_protocols(icmp.icmp)

        if (eth.ethertype == ether_types.ETH_TYPE_LLDP) or (len(IPV6) != 0):
            # ignore lldp packet
            return
        else:
            req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
            datapath.send_msg(req)
        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        if len(IPV4) != 0:
            self.total_length = IPV4[0].total_length
            self.flags = IPV4[0].flags
            self.csum = IPV4[0].csum
            self.src_ip = IPV4[0].src

        if len(UDP) != 0:
            self.src_port = UDP[0].src_port

        if len(TCP) != 0:
            self.src_port = TCP[0].src_port


        # self.logger.info("packet in %s %s %s %s", dpid, src, dst, self.in_port)

        # # learn a mac address to avoid FLOOD next time.
        # self.mac_to_port[dpid][src] = self.in_port
        #
        # if dst in self.mac_to_port[dpid]:
        #     out_port = self.mac_to_port[dpid][dst]
        # else:
        #     out_port = ofproto.OFPP_FLOOD
        #
        # actions = [parser.OFPActionOutput(out_port)]
        #
        # # install a flow to avoid packet_in next time
        # if out_port != ofproto.OFPP_FLOOD:
        #     match = parser.OFPMatch(in_port=self.in_port, eth_dst=dst, eth_src=src)
        #     # verify if we have a valid buffer_id, if yes avoid to send both
        #     # flow_mod & packet_out
        #     if msg.buffer_id != ofproto.OFP_NO_BUFFER:
        #         # self.add_flow(datapath, 1, match, actions, msg.buffer_id)
        #         return
        #     else:
        #         return
                # self.add_flow(datapath, 1, match, actions)
        # data = None
        # if msg.buffer_id == ofproto.OFP_NO_BUFFER:
        #     data = msg.data

        # out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id, in_port=self.in_port, actions=actions, data=data)
        # datapath.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        self.logger.info(self.i)
        for stat in sorted(body, key=attrgetter("port_no")):
            if stat.port_no == int(self.in_port):
                # print("test")
                f1 = open("svmrbf", "a+")
                f2 = open("knn", "a+")
                f3 = open("dtc", "a+")
                f4 = open("rfc", "a+")
                f5 = open("mlp", "a+")
                f6 = open("adb", "a+")
                f7 = open("gnb", "a+")
                f8 = open("svmlin", "a+")
                data = np.array([[flags.transform([int(self.flags)])[0],
                                    src_ip.transform([(self.src_ip)])[0]]])

                # print(data)
                # data = le.fit_transform(data)
                data = scaler.transform(data)
                # data = data.transpose()
                # print(data)
                res1 = self.svmrbf.predict(data)
                res2 = self.knn.predict(data)
                res3 = self.dtc.predict(data)
                res4 = self.rfc.predict(data)
                res5 = self.mlp.predict(data)
                res6 = self.adb.predict(data)
                res7 = self.gnb.predict(data)
                res8 = self.svmlin.predict(data)
                self.i = self.i + 1
                # print(self.i)
                # print(res1[0])

                f1.write(str(self.total_length)+";"+
                str(self.flags)+";"+
                str(self.csum)+";"+
                str(self.src_ip)+";"+
                str(self.src_port)+";"+
                str(stat.port_no)+";"+str(res1[0]))
                f1.write("\n")
                # f1.close()

                f2.write(str(self.total_length)+";"+
                str(self.flags)+";"+
                str(self.csum)+";"+
                str(self.src_ip)+";"+
                str(self.src_port)+";"+
                str(stat.port_no)+";"+str(res2[0]))
                f2.write("\n")

                f3.write(str(self.total_length)+";"+
                str(self.flags)+";"+
                str(self.csum)+";"+
                str(self.src_ip)+";"+
                str(self.src_port)+";"+
                str(stat.port_no)+";"+str(res3[0]))
                f3.write("\n")
                #f3.close()

                f4.write(str(self.total_length)+";"+
                str(self.flags)+";"+
                str(self.csum)+";"+
                str(self.src_ip)+";"+
                str(self.src_port)+";"+
                str(stat.port_no)+";"+str(res4[0]))
                f4.write("\n")
                #f4.close()

                f5.write(str(self.total_length)+";"+
                str(self.flags)+";"+
                str(self.csum)+";"+
                str(self.src_ip)+";"+
                str(self.src_port)+";"+
                str(stat.port_no)+";"+str(res5[0]))
                f5.write("\n")
                #f5.close()

                f6.write(str(self.total_length)+";"+
                str(self.flags)+";"+
                str(self.csum)+";"+
                str(self.src_ip)+";"+
                str(self.src_port)+";"+
                str(stat.port_no)+";"+str(res6[0]))
                f6.write("\n")
                #f6.close()

                f7.write(str(self.total_length)+";"+
                str(self.flags)+";"+
                str(self.csum)+";"+
                str(self.src_ip)+";"+
                str(self.src_port)+";"+
                str(stat.port_no)+";"+str(res7[0]))
                f7.write("\n")
                #f7.close()

                f8.write(str(self.total_length)+";"+
                str(self.flags)+";"+
                str(self.csum)+";"+
                str(self.src_ip)+";"+
                str(self.src_port)+";"+
                str(stat.port_no)+";"+str(res8[0]))
                f8.write("\n")
                #f9.close()
