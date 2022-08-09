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


filepath1 = "/home/fauzi/low rate attack/traindataset.csv"

filepath2 = "/home/fauzi/low rate attack/testdataset.csv"

train = pd.read_csv(filepath1)
test = pd.read_csv(filepath2)

train = train[train.notnull()]
test = test[test.notnull()]

train.head()
test.head()

feat_labels = list(train.columns)
print(feat_labels)

print(len(train))
print(len(test))

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
spec = preprocessing.LabelEncoder()

datapath_id	= preprocessing.LabelEncoder()
version	= preprocessing.LabelEncoder()
header_length = preprocessing.LabelEncoder()
tos	= preprocessing.LabelEncoder()
total_length = preprocessing.LabelEncoder()
flags = preprocessing.LabelEncoder()
offset = preprocessing.LabelEncoder()
ttl = preprocessing.LabelEncoder()
proto = preprocessing.LabelEncoder()
csum = preprocessing.LabelEncoder()
src_ip = preprocessing.LabelEncoder()
dst_ip = preprocessing.LabelEncoder()
src_port = preprocessing.LabelEncoder()
dst_port = preprocessing.LabelEncoder()
port_no = preprocessing.LabelEncoder()
rx_bytes_ave = preprocessing.LabelEncoder()
rx_error_ave = preprocessing.LabelEncoder()
rx_dropped_ave = preprocessing.LabelEncoder()
tx_bytes_ave = preprocessing.LabelEncoder()
tx_error_ave = preprocessing.LabelEncoder()
tx_dropped_ave = preprocessing.LabelEncoder()


X_train = train.drop(columns=['label'])
# print(y_train)

X_test = test.drop(columns=['label'])
# print(y_test)

X_trainnew = train.drop(columns=['label'])
# print(y_train)

X_testnew = test.drop(columns=['label'])
# print(y_test)

X_trainnew = X_trainnew.apply(le.fit_transform)
X_testnew = X_testnew.apply(le.fit_transform)


# datapath_id.fit(X_train['datapath_id'])
# version.fit(X_train['version'])
# header_length.fit(X_train['header_length'])
# tos.fit(X_train['tos'])
# total_length.fit(X_train['total_length'])
# flagss.fit(X_train['flags'])
# print(flagss.classes_)
# offset.fit(X_train['offset'])
# ttl.fit(X_train['ttl'])
# proto.fit(X_train['proto'])
# csum.fit(X_train['csum'])
# src_ip.fit(X_train['src_ip'])
# dst_ip.fit(X_train['dst_ip'])
# src_port.fit(X_train['src_port'])
# dst_port.fit(X_train['dst_port'])
# tcp_flag.fit(X_train['tcp_flag'])
# type_icmp.fit(X_train['type_icmp'])
# code_icmp.fit(X_train['code_icmp'])
# csum_icmp.fit(X_train['csum_icmp'])
# port_no.fit(X_train['port_no'])
# rx_bytes_ave.fit(X_train['rx_bytes_ave'])
# rx_error_ave.fit(X_train['rx_error_ave'])
# rx_dropped_ave.fit(X_train['rx_dropped_ave'])
# tx_bytes_ave.fit(X_train['tx_bytes_ave'])
# tx_error_ave.fit(X_train['tx_error_ave'])
# tx_dropped_ave.fit(X_train['tx_dropped_ave'])

datapath_id.fit(X_test['datapath_id'])
version.fit(X_test['version'])
header_length.fit(X_test['header_length'])
tos.fit(X_test['tos'])
total_length.fit(X_test['total_length'])
flags.fit(X_test['flags'])
offset.fit(X_test['offset'])
ttl.fit(X_test['ttl'])
proto.fit(X_test['proto'])
csum.fit(X_test['csum'])
src_ip.fit(X_test['src_ip'])
dst_ip.fit(X_test['dst_ip'])
src_port.fit(X_test['src_port'])
dst_port.fit(X_test['dst_port'])
port_no.fit(X_test['port_no'])
rx_bytes_ave.fit(X_test['rx_bytes_ave'])
rx_error_ave.fit(X_test['rx_error_ave'])
rx_dropped_ave.fit(X_test['rx_dropped_ave'])
tx_bytes_ave.fit(X_test['tx_bytes_ave'])
tx_error_ave.fit(X_test['tx_error_ave'])
tx_dropped_ave.fit(X_test['tx_dropped_ave'])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_important_train = scaler.fit_transform(X_trainnew)
X_important_test = scaler.fit_transform(X_testnew)






class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}

        # IPV4
        self.version = ""
        self.header_length = ""
        self.tos = ""
        self.total_length = ""
        self.flags = ""
        self.offset = ""
        self.ttl = ""
        self.proto = ""
        self.csum = ""
        self.src_ip = ""
        self.dst_ip = ""
        self.i = 0
        # TCP & UDP
        self.src_port = "0"
        self.dst_port = "0"


        self.filename6 = "/home/fauzi/low rate attack/model/svmrbf.sav"
        self.filename7 = "/home/fauzi/low rate attack/model/knn.sav"
        self.filename8 = "/home/fauzi/low rate attack/model/dtc.sav"
        self.filename9 = "/home/fauzi/low rate attack/model/rfc.sav"
        self.filename10 = "/home/fauzi/low rate attack/model/mlp.sav"
        self.filename11 = "/home/fauzi/low rate attack/model/adc.sav"
        self.filename12 = "/home/fauzi/low rate attack/model/gnb.sav"
        self.filename14 = "/home/fauzi/low rate attack/model/svmlin.sav"

        self.svmrbf = pickle.load(open(self.filename6, 'rb'))
        self.knn = pickle.load(open(self.filename7, 'rb'))
        self.dtc = pickle.load(open(self.filename8, 'rb'))
        self.rfc = pickle.load(open(self.filename9, 'rb'))
        self.mlp = pickle.load(open(self.filename10, 'rb'))
        self.adc = pickle.load(open(self.filename11, 'rb'))
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
            self.version = IPV4[0].version
            self.header_length = IPV4[0].header_length
            self.tos = IPV4[0].tos
            self.total_length = IPV4[0].total_length
            self.flags = IPV4[0].flags
            self.offset = IPV4[0].offset
            self.ttl = IPV4[0].ttl
            self.proto = IPV4[0].proto
            self.csum = IPV4[0].csum
            self.src_ip = IPV4[0].src
            self.dst_ip = IPV4[0].dst

        if len(UDP) != 0:
            self.src_port = UDP[0].src_port
            self.dst_port = UDP[0].dst_port

        if len(TCP) != 0:
            self.src_port = TCP[0].src_port
            self.dst_port = TCP[0].dst_port
            self.tcp_flag = TCP[0].bits

        if len(ICMP) != 0:
            self.type_icmp = ICMP[0].type
            self.code_icmp = ICMP[0].code
            self.csum_icmp = ICMP[0].csum

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
                f6 = open("adc", "a+")
                f7 = open("gnb", "a+")
                f9 = open("svmlin", "a+")
                data = np.array([[datapath_id.transform([(ev.msg.datapath.id)])[0],
                                    version.transform([(self.version)])[0],
                                    header_length.transform([(self.header_length)])[0],
                                    tos.transform([(self.tos)])[0],
                                    total_length.transform([(self.total_length)])[0],
                                    flags.transform([int(self.flags)])[0],
                                    offset.transform([(self.offset)])[0],
                                    ttl.transform([(self.ttl)])[0],
                                    proto.transform([(self.proto)])[0],
                                    csum.transform([(self.csum)])[0],
                                    src_ip.transform([(self.src_ip)])[0],
                                    dst_ip.transform([(self.dst_ip)])[0],
                                    src_port.transform([(self.src_port)])[0],
                                    dst_port.transform([(self.dst_port)])[0],
                                    port_no.transform([(stat.port_no)])[0],
                                    spec.fit_transform([((stat.rx_bytes / stat.rx_packets))])[0],
                                    rx_error_ave.transform([((stat.rx_errors / stat.rx_packets))])[0],
                                    rx_dropped_ave.transform([((stat.rx_dropped / stat.rx_packets))])[0],
                                    spec.fit_transform([((stat.tx_bytes / stat.tx_packets))])[0],
                                    tx_error_ave.transform([((stat.tx_errors / stat.tx_packets))])[0],
                                    tx_dropped_ave.transform([((stat.tx_dropped / stat.tx_packets))])[0]]])
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
                res6 = self.adc.predict(data)
                res7 = self.gnb.predict(data)
                res9 = self.svmlin.predict(data)
                self.i = self.i + 1
                # print(self.i)
                # print(res1[0])

                # str(ev.msg.datapath.id)+";"+
                # str(self.version)+";"+
                # str(self.header_length)+";"+
                # str(self.tos)+";"+
                # str(self.total_length)+";"+
                # str(self.flags)+";"+
                # str(self.offset)+";"+
                # str(self.ttl)+";"+
                # str(self.proto)+";"+
                # str(self.csum)+";"+
                # str(self.src_ip)+";"+
                # str(self.dst_ip)+";"+
                # str(self.src_port)+";"+
                # str(self.dst_port)+";"+
                # str(self.type_icmp)+";"+
                # str(self.tcp_flag)+";"+
                # str(self.code_icmp)+";"+
                # str(self.csum_icmp)+";"+
                # str(stat.port_no)+";"+

                f1.write(str(ev.msg.datapath.id)+";"+
                str(self.version)+";"+
                str(self.header_length)+";"+
                str(self.tos)+";"+
                str(self.total_length)+";"+
                str(self.flags)+";"+
                str(self.offset)+";"+
                str(self.ttl)+";"+
                str(self.proto)+";"+
                str(self.csum)+";"+
                str(self.src_ip)+";"+
                str(self.dst_ip)+";"+
                str(self.src_port)+";"+
                str(self.dst_port)+";"+
                str(stat.port_no)+";"+str(res1[0]))
                f1.write("\n")
                # f1.close()

                f2.write(str(ev.msg.datapath.id)+";"+
                str(self.version)+";"+
                str(self.header_length)+";"+
                str(self.tos)+";"+
                str(self.total_length)+";"+
                str(self.flags)+";"+
                str(self.offset)+";"+
                str(self.ttl)+";"+
                str(self.proto)+";"+
                str(self.csum)+";"+
                str(self.src_ip)+";"+
                str(self.dst_ip)+";"+
                str(self.src_port)+";"+
                str(self.dst_port)+";"+
                str(stat.port_no)+";"+str(res2[0]))
                f2.write("\n")
                #f2.close()

                f3.write(str(ev.msg.datapath.id)+";"+
                    str(self.version)+";"+
                    str(self.header_length)+";"+
                    str(self.tos)+";"+
                    str(self.total_length)+";"+
                    str(self.flags)+";"+
                    str(self.offset)+";"+
                    str(self.ttl)+";"+
                    str(self.proto)+";"+
                    str(self.csum)+";"+
                    str(self.src_ip)+";"+
                    str(self.dst_ip)+";"+
                    str(self.src_port)+";"+
                    str(self.dst_port)+";"+
                    str(stat.port_no)+";"+str(res3[0]))
                f3.write("\n")
                #f3.close()

                f4.write(str(ev.msg.datapath.id)+";"+
                    str(self.version)+";"+
                    str(self.header_length)+";"+
                    str(self.tos)+";"+
                    str(self.total_length)+";"+
                    str(self.flags)+";"+
                    str(self.offset)+";"+
                    str(self.ttl)+";"+
                    str(self.proto)+";"+
                    str(self.csum)+";"+
                    str(self.src_ip)+";"+
                    str(self.dst_ip)+";"+
                    str(self.src_port)+";"+
                    str(self.dst_port)+";"+
                    str(stat.port_no)+";"+str(res4[0]))
                f4.write("\n")
                #f4.close()

                f5.write(str(ev.msg.datapath.id)+";"+
                    str(self.version)+";"+
                    str(self.header_length)+";"+
                    str(self.tos)+";"+
                    str(self.total_length)+";"+
                    str(self.flags)+";"+
                    str(self.offset)+";"+
                    str(self.ttl)+";"+
                    str(self.proto)+";"+
                    str(self.csum)+";"+
                    str(self.src_ip)+";"+
                    str(self.dst_ip)+";"+
                    str(self.src_port)+";"+
                    str(self.dst_port)+";"+
                    str(stat.port_no)+";"+str(res5[0]))
                f5.write("\n")
                #f5.close()

                f6.write(str(ev.msg.datapath.id)+";"+
                    str(self.version)+";"+
                    str(self.header_length)+";"+
                    str(self.tos)+";"+
                    str(self.total_length)+";"+
                    str(self.flags)+";"+
                    str(self.offset)+";"+
                    str(self.ttl)+";"+
                    str(self.proto)+";"+
                    str(self.csum)+";"+
                    str(self.src_ip)+";"+
                    str(self.dst_ip)+";"+
                    str(self.src_port)+";"+
                    str(self.dst_port)+";"+
                    str(stat.port_no)+";"+str(res6[0]))
                f6.write("\n")
                #f6.close()

                f7.write(str(ev.msg.datapath.id)+";"+
                    str(self.version)+";"+
                    str(self.header_length)+";"+
                    str(self.tos)+";"+
                    str(self.total_length)+";"+
                    str(self.flags)+";"+
                    str(self.offset)+";"+
                    str(self.ttl)+";"+
                    str(self.proto)+";"+
                    str(self.csum)+";"+
                    str(self.src_ip)+";"+
                    str(self.dst_ip)+";"+
                    str(self.src_port)+";"+
                    str(self.dst_port)+";"+
                    str(stat.port_no)+";"+str(res7[0]))
                f7.write("\n")
                #f7.close()

                f9.write(str(ev.msg.datapath.id)+";"+
                        str(self.version)+";"+
                        str(self.header_length)+";"+
                        str(self.tos)+";"+
                        str(self.total_length)+";"+
                        str(self.flags)+";"+
                        str(self.offset)+";"+
                        str(self.ttl)+";"+
                        str(self.proto)+";"+
                        str(self.csum)+";"+
                        str(self.src_ip)+";"+
                        str(self.dst_ip)+";"+
                        str(self.src_port)+";"+
                        str(self.dst_port)+";"+
                        str(stat.port_no)+";"+str(res9[0]))
                f9.write("\n")
                #f9.close()
