'''
==================================================================
Frame structure environemnt file
This file contains environment for train agent using reinforcement learning
    ENV    : contains Game class
    Game1 : REDUCE TOTAL SURFACE for Q-Learning , Double Q-Learning and Actor-Critic - 6 Actions
    Game2 : REDUCE STRAIN ENERGY for Q-Learning , Double Q-Learning and Actor-Critic - 6 Actions
    Game3 : REDUCE TOTAL SURFACE for DDPG - 6 Actions
    Game4 : REDUCE STRAIN ENERGY for DDPG - 6 Actions
    Game5 : REDUCE TOTAL SURFACE for MADDPG - 6 Actions
    Game6 : REDUCE STRAIN ENERGY for MADDPG - 6 Actions

Adjustable parameter are under '研究室'

フレーム構造環境ファイル
このファイルには、強化学習を使用したトレーニングエージェントの環境が含まれています
    ENV：ゲームクラスを含む
    Game1 : Q-Learning、Double Q-Learning、Actor-Criticの総表面を減らす-6行動
    Game2 : Q-Learning、Double Q-Learning、Actor-Criticの歪みエネルギーの削減-6行動
    Game3 : DDPGの総表面を減らす-6行動
    Game4 : DDPGの歪みエネルギーの削減-6行動
    Game5 : MADDPGの総表面を減らす-6行動
    Game6 : MADDPGの歪みエネルギーの削減-6行動
調整可能なパラメータは「研究室」の下にあります
==================================================================
'''

'''
======================================================
IMPORT PART
======================================================
'''
import math
import os
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



'''
======================================================
Helper functions
======================================================
'''
def state_data(i,j,self):
    # Boundary Conditions ----------------------------
    bc = 0
    if (self.gen_model.n_u_name_div[i][j].res) == [0,0,0,0,0,0]:
        bc = 1 # free
    elif (self.gen_model.n_u_name_div[i][j].res) == [0,1,0,0,0,0]:
        bc = 2 # roller x, z
    elif (self.gen_model.n_u_name_div[i][j].res) == [1,1,1,0,0,0]:
        bc = 3 # pin
    elif (self.gen_model.n_u_name_div[i][j].res) == [1,1,1,1,1,1]:
        bc = 4 # fixed
    bc = bc/4

    # Theta --------------------------------------------
    try:
        if i-1 >= 0:
            n_up = (self.gen_model.n_u_name_div[i][j].coord[1] - self.gen_model.n_u_name_div[i-1][j].coord[1])/self.gen_model.span
        else:
            n_up = 0
    except:
        n_up = 0
    try:
        n_down = (self.gen_model.n_u_name_div[i][j].coord[1] - self.gen_model.n_u_name_div[i+1][j].coord[1])/self.gen_model.span
    except:
        n_down = 0
    try:
        if j-1 >= 0:
            n_left = (self.gen_model.n_u_name_div[i][j].coord[1] - self.gen_model.n_u_name_div[i][j-1].coord[1])/self.gen_model.span
        else:
            n_left = 0
    except:
        n_left = 0
    try:
        n_right = (self.gen_model.n_u_name_div[i][j].coord[1] - self.gen_model.n_u_name_div[i][j+1].coord[1])/self.gen_model.span
    except:
        n_right = 0

    # Rotation-x-i ------------------------------------
    try:
        dtxi = (self.gen_model.n_u_name_div[i][j].global_d[3][0]-self.dtxmin)/(self.dtxmax-self.dtxmin)
    except:
        dtxi = 0

    # Rotation-z-i ------------------------------------
    try:
        dtzi = (self.gen_model.n_u_name_div[i][j].global_d[5][0]-self.dtzmin)/(self.dtzmax-self.dtzmin)
    except:
        dtzi =0
    # Rotation-x-j ------------------------------------
    try:
        if i-1 >= 0:
            dtxj_up = (self.gen_model.n_u_name_div[i-1][j].global_d[3][0]-self.dtxmin)/(self.dtxmax-self.dtxmin)
        else:
            dtxj_up = 0
    except:
        dtxj_up = 0
    try:
        dtxj_down = (self.gen_model.n_u_name_div[i+1][j].global_d[3][0]-self.dtxmin)/(self.dtxmax-self.dtxmin)
    except:
        dtxj_down = 0
    try:
        if j-1 >= 0:
            dtxj_left = (self.gen_model.n_u_name_div[i][j-1].global_d[3][0]-self.dtxmin)/(self.dtxmax-self.dtxmin)
        else:
            dtxj_left = 0
    except:
        dtxj_left = 0
    try:
        dtxj_right = (self.gen_model.n_u_name_div[i][j+1].global_d[3][0]-self.dtxmin)/(self.dtxmax-self.dtxmin)
    except:
        dtxj_right = 0

    # Rotation-z-j ------------------------------------
    try:
        if i-1 >= 0:
            dtzj_up = (self.gen_model.n_u_name_div[i-1][j].global_d[5][0]-self.dtzmin)/(self.dtzmax-self.dtzmin)
        else:
            dtzj_up = 0
    except:
        dtzj_up = 0
    try:
        dtzj_down = (self.gen_model.n_u_name_div[i+1][j].global_d[5][0]-self.dtzmin)/(self.dtzmax-self.dtzmin)
    except:
        dtzj_down = 0
    try:
        if j-1 >= 0:
            dtzj_left = (self.gen_model.n_u_name_div[i][j-1].global_d[5][0]-self.dtzmin)/(self.dtzmax-self.dtzmin)
        else:
            dtzj_left = 0
    except:
        dtzj_left = 0
    try:
        dtzj_right = (self.gen_model.n_u_name_div[i][j+1].global_d[5][0]-self.dtzmin)/(self.dtzmax-self.dtzmin)
    except:
        dtzj_right = 0

    # Deformation-i ------------------------------------
    try:
        di = (self.gen_model.n_u_name_div[i][j].global_d[1][0]-self.dymin)/(self.dymax-self.dymin)
    except:
        di = 0
    # Deformation-j ------------------------------------
    try:
        if i-1 >= 0:
            dj_up = (self.gen_model.n_u_name_div[i-1][j].global_d[1][0]-self.dymin)/(self.dymax-self.dymin)
        else:
            dj_up = 0
    except:
        dj_up = 0
    try:
        dj_down = (self.gen_model.n_u_name_div[i+1][j].global_d[1][0]-self.dymin)/(self.dymax-self.dymin)
    except:
        dj_down = 0
    try:
        if j-1 >= 0:
            dj_left = (self.gen_model.n_u_name_div[i][j-1].global_d[1][0]-self.dymin)/(self.dymax-self.dymin)
        else:
            dj_left = 0
    except:
        dj_left = 0
    try:
        dj_right = (self.gen_model.n_u_name_div[i][j+1].global_d[1][0]-self.dymin)/(self.dymax-self.dymin)
    except:
        dj_right = 0

    # Zi/Zmax -------------------------------------------
    geo = (self.gen_model.n_u_name_div[i][j].coord[1])/(self.max_y_val)
    # PosXi/PosXmax -------------------------------------
    pos1= i/self.num_x
    # PosYi/PosYmax -------------------------------------
    pos2= j/self.num_z

    return dtxi,dtzi,dtxj_up,dtxj_down,dtxj_left,dtxj_right,dtzj_up,dtzj_down,dtzj_left,dtzj_right,di,dj_up,dj_down,dj_left,dj_right,n_up,n_down,n_left,n_right,pos1,pos2,bc,geo


'''
======================================================
CLASS PART
======================================================
'''
# Environment class, contain Game class. Do not change / 環境クラス、ゲームクラスを含みます。 変えないで
class ENV:
    def __init__(self,game):
        self.name = 'FRAME_ENV'
        self.game = game
        self.num_agents = game.num_agents
        self.over = 0
        #=======================
        #State Action Reward Next_State Done
        #=======================
        self.state = self.game.state
        self.action = self.game.action #get action from rl or data file
        self.reward = self.game.reward
        self.next_state = self.game.next_state
        self.done = self.game.done
        #=======================
        #Output Action
        #=======================
        self.output = [] #output = [St,at,rt,St+1,Done]

    def check_over(self):
        if self.game.done_counter == 1:
            self.over = 1
        else:
            pass

    def reset(self):
        self.over = 0
        self.game.reset()
        self.state = self.game.state
        self.action = self.game.action
        self.reward = self.game.reward
        self.next_state = self.game.next_state
        self.done = self.game.done
        self.output = []

    def gen_output(self):
        '''
        Output_list
        1. replay buffer = [St,at,rt,St+1,Done]
        2. other format(.txt)
        3. render
        '''
        # reset output to empty list
        for i in range(self.num_agents):
            x = []
            # output = [St,at,rt,St+1,Done] one replay buffer
            x.append(self.state[-1])
            x.append(self.action[-1])
            x.append(self.reward[-1])
            x.append(self.next_state[-1])
            x.append(self.done[-1])
            self.output.append(x)

    def save_output(self):
        # save replaybufferfile as txt, csv
        pass

#=============================================================================
# GAME 1 '研究室'
class Game1:
    def __init__(self,end_step,alpha,max_y_val,model,num_agents=1,render=0,tell_action=False):
        self.name = 'GAME 1' # Name of the game / ゲームの名前
        self.description = 'AGENT HAS 6 ACTIONS:  MOVE NODE (UP DOWN), MOVE TO SURROUNDING NODES (LEFT RIGHT UP DOWN)' # Game's description / ゲームの説明
        self.objective = 'REDUCE TOTAL SURFACE' # Game's objective / ゲームの目的
        self.tell_action =tell_action # Print agent action in console /コンソールでエージェントの行動を印刷する
        self.num_agents = num_agents # Amount of agents in the game / ゲーム内のエージェントの数
        self.gen_model = model # Gen structural model used in the game / ゲームで使用されるGen構造モデル
        self.model = model.model # Structural model used in the game / ゲームで使用される構造モデル
        self.num_x = model.num_x # Amount of Structural model's span in x axis (horizontal) / X軸での構造モデルのスパンの量（水平）
        self.num_z = model.num_z # Amount of Structural model's span in z axis (horizontal) / Z軸での構造モデルのスパンの量（水平）
        self.render = render # Render after each step / 各ステップ後にレンダリング
        self.game_step = 1 # Game initial step / ゲームの最初のステップ
        self.game_type = 0 # Game's state type / ゲームの状態タイプ
        self.end_step = end_step # Game final step / ゲームの最終ステップ
        self.alpha = alpha # Magnitude for agents to adjust structure as a factor of Structural model's span / エージェントが構造モデルのスパンの要素として構造を調整するための大きさ
        self.y_step = self.alpha*self.gen_model.span # Magnitude for agents to adjust structure(m) / エージェントが構造を調整するための大きさ（m）

        self.state = [] # Game state / ゲームの状態
        self.action = [] # Game action / ゲームの行動
        self.reward = [] # Game reward for each agent / 各エージェントのゲーム報酬
        self.next_state = [] # Game next state / ゲームの次の状態
        self.done = [] # Game over counter / ゲームオーバーカウンター

        self.doing = [] # List of position(x,z) in the structure of each agent / 各エージェントの構造内のposition（x、z）のリスト
        for i in range(self.num_agents): # Initialize starting position of each structure / 各構造の開始位置を初期化
            self.doing.append([0,0])
        self.metagrid = [] # 2-D Array of data in each structural node / 各構造ノードのデータの2次元配列
        for i in range(self.num_z): # Initialize structural data array / 構造データ配列を初期化する
            self.metagrid.append([])
            for j in range(self.num_x):
                self.metagrid[-1].append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) # Maximum num_state in this suite is 23 / このスイートの最大num_stateは23です
        self.xmax = 0 # Maximum x coordinate value in this structural model (horizontal) / この構造モデルの最大x座標値（水平）
        self.xmin = 0 # Minimum x coordinate value in this structural model (horizontal) / この構造モデル（水平）の最小x座標値
        self.ymax = 0 # Maximum y coordinate value in this structural model (vertical) / この構造モデルのY座標の最大値（垂直）
        self.ymin = 0 # Minimum y coordinate value in this structural model (vertical) / この構造モデルのY座標の最小値（垂直）
        self.zmax = 0 # Maximum z coordinate value in this structural model (horizontal) / この構造モデルの最大Z座標値（水平）
        self.zmin = 0 # Minimum z coordinate value in this structural model (horizontal) / この構造モデルの最小Z座標値（水平）
        self.sval = 0.001 # small noise / 小さなノイズ

        self.dymax = 0
        self.dymin = 0
        self.sdyval = 0
        self.dmax0 = 0
        self.dmaxset = 0
        self.dtxmax = 0
        self.dtxmin = 0
        self.dtymax = 0
        self.dtymin = 0
        self.dtzmax = 0
        self.dtzmin = 0
        self.dtxmax0 = 0
        self.dtymax0 = 0
        self.dtzmax0 = 0

        self.max_y_val = max_y_val # Maximum y coordinate value in this structural model for this game (vertical) / このゲームのこの構造モデルの最大y座標値（垂直）
        #**********
        self.int_strain_e = 0 # Initial Total length for this game / このゲームの初期の全長
        self.strain_e = 0 # Current Total length  for this game / このゲームの現在の合計の長さ
        self.next_strain_e = 0 # Total length  after agents do actions. Used for calculating reward / エージェントがアクションを実行した後の全長。 報酬の計算に使用されます
        #**********
        self.reward_counter = [] # List of reward of each agent / 各エージェントの報酬一覧
        for i in range(self.num_agents): # Initialize reward for each agent / 各エージェントの報酬を初期化する
            self.reward.append(0)
            self.reward_counter.append(0)
        self.done_counter = 0 # Counter for game end / ゲーム終了のカウンター

        # Print out game properties at beginning of the game / ゲームの開始時にゲームのプロパティを印刷する
        print('-------------------------------------------------------')
        print(self.description)
        print(self.objective)
        print('GAME WILL BE ENDED AFTER {} STEP'.format(self.end_step))
        print('-------------------------------------------------------')

    # Function to set Game's state type / ゲームの状態タイプを設定する関数
    def set_gametype(self,game_type):
        self.game_type = game_type

    # Fuction to update structural data array / 構造化データ配列を更新する関数
    def _update_metagrid(self):
        # update structure
        self.model.restore()
        self.model.gen_all()

        xlist = []
        ylist = []
        zlist = []
        dylist = []
        dtxlist = []
        dtylist = []
        dtzlist = []

        for i in range(len(self.model.nodes)):
            xlist.append(self.model.nodes[i].coord[0])
            ylist.append(self.model.nodes[i].coord[1])
            zlist.append(self.model.nodes[i].coord[2])
            dylist.append(self.model.nodes[i].global_d[1][0])
            dtxlist.append(self.model.nodes[i].global_d[3][0])
            dtylist.append(self.model.nodes[i].global_d[4][0])
            dtzlist.append(self.model.nodes[i].global_d[5][0])

        self.xmax = max(xlist)
        self.xmin = min(xlist)
        self.ymax = max(ylist)
        self.ymin = min(ylist)
        self.zmax = max(zlist)
        self.zmin = min(zlist)
        self.sdyval = self.sval*self.dymin
        self.dtxmax = max(dtxlist)
        self.dtxmin = min(dtxlist)
        self.dtymax = max(dtylist)
        self.dtymin = min(dtylist)
        self.dtzmax = max(dtzlist)
        self.dtzmin = min(dtzlist)
        self.dymax = max(dylist)
        self.dymin = min(dylist)

        if self.dmaxset == 0:
            self.dmax0 = abs(min(dylist))
            self.dtxmax0 = max([abs(min(dtxlist)),abs(max(dtxlist))])
            self.dtymax0 = max([abs(min(dtylist)),abs(max(dtylist))])
            self.dtzmax0 = max([abs(min(dtzlist)),abs(max(dtzlist))])

            self.dmaxset = 1

        for i in range(self.num_z):
            for j in range(self.num_x):
                dtxi,dtzi,dtxj_up,dtxj_down,dtxj_left,dtxj_right,dtzj_up,dtzj_down,dtzj_left,dtzj_right,di,dj_up,dj_down,dj_left,dj_right,n_up,n_down,n_left,n_right,pos1,pos2,bc,geo = state_data(i,j,self)
                '''
                self.metagrid[i][j] = [[dtxi],[dtzi],[dtxj_up],[dtxj_down],[dtxj_left],[dtxj_right],[dtzj_up],[dtzj_down],[dtzj_left],[dtzj_right],[di],[dj_up],[dj_down],[dj_left],[dj_right],[n_up],[n_down],[n_left],[n_right],[pos1],[pos2],[bc],[geo]]
                '''
                self.metagrid[i][j] = [
                [n_up],[n_down],[n_left],[n_right],
                [pos1],[pos2],
                [geo]
                ]

    # Function to calculate value use in the reward system / 報酬制度での価値利用を計算する機能
    def _game_gen_state_condi(self):
        self.gen_model.gen_surface1() # Calculate total surface / 総表面積を計算する
        self.strain_e = self.gen_model.surface_1 # Current total surface of this structure / この構造の総表面積
        if self.game_step == 1: # Initial total length of this structure / この構造の初期の全長
            self.int_strain_e = self.gen_model.surface_1
        else:
            pass

    # Function to initialize state / 状態を初期化する関数
    def _game_get_1_state(self,do):
        self._update_metagrid() # update structural data array / 構造データ配列を更新する
        # do = [i,j]
        # metagrid[z,x]
        # Check game type to generate state from structural data array / 構造データ配列から状態を生成するゲームタイプをチェックしてください
        x = self.metagrid[do[0]][do[1]]
        state = np.array(x) # state / 状態
        return state

    # Function to generate next state / 次の状態を生成する関数
    def _game_get_next_state(self,do,action,i=0):
        num = [action[0][0],action[0][1],action[0][2],action[0][3],action[0][4],action[0][5]]
        num = num.index(max(num)) # Find maximum index of  the action receive from Neural Network / ニューラルネットワークから受け取るアクションの最大インデックスを見つける
        # next_state = f(action)
        # Interpretation of action index / 行動のインデックスの解釈
        if num == 0: # Adjust this node by moving up in the magnitude of y_step / y_stepの大きさを上に移動して、このノードを調整します
            if self.gen_model.n_u_name_div[do[0]][do[1]].res[1] !=1:
                if self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] + self.y_step <= self.max_y_val:
                    self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] += self.y_step
                else:
                    pass
            else:
                pass
        elif num == 1: # Adjust this node by moving down in the magnitude of y_step / y_stepの大きさを下に移動して、このノードを調整します
            if self.gen_model.n_u_name_div[do[0]][do[1]].res[1] != 1:
                if self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] - self.y_step >= 0:
                    self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] -= self.y_step
                else:
                    pass
            else:
                pass
        elif num == 2: # Agent move to other node to the right(move right x+1) / ＃エージェントは他のノードに右に移動します（右にx + 1移動）
            # do[z,x]
            if (do[1]+1 != (len(self.gen_model.n_u_name_div[0]))):
                self.doing[0][1] = do[1]+1
            else:
                pass
        elif num == 3: # Agent move to other node to the left(move left x-1) / エージェントは他のノードに左に移動します（左に移動x-1）
            # do[z,x]
            if (do[1] != 0):
                self.doing[0][1] = do[1]-1
            else:
                pass
        elif num == 4: # Agent move to other node to the upper(move up z-1) / エージェントが他のノードに移動します（上に移動z-1）
            # do[z,x]
            if (do[0] != 0):
                self.doing[0][0] = do[0]-1
            else:
                pass
        elif num == 5: # Agent move to other node to the lower(move down z+1) / エージェントは他のノードに移動します（z + 1に移動）
            # do[z,x]
            if (do[0]+1 != (len(self.gen_model.n_u_name_div))):
                self.doing[0][0] = do[0]+1
            else:
                pass
        announce = ['z_up','z_down','move right','move left','move up','move down'] # list of actions / 行動のリスト
        if self.tell_action == True:
            print(announce[num-1]) # print out action if tell_action is Trues / tell_actionがTrueの場合、行動を出力します
        self._update_metagrid()  # update structural data array / 構造データ配列を更新する
        # Check game type to generate state from structural data array / 構造データ配列から状態を生成するゲームタイプをチェックしてください
        x = self.metagrid[self.doing[i][0]][self.doing[i][1]]
        next_state = np.array(x) # next_state / 次の状態
        return next_state

    # Function to calculate value use in the reward system / 報酬制度での価値利用を計算する機能
    def _gen_gen_reward_condition(self):

        self.gen_model.gen_surface1() # Calculate next state's total surface / 次の状態の総表面積を計算する
        self.next_strain_e = self.gen_model.surface_1# Total surface of this structure in the next_state after agents do actions / エージェントが行動を実行した後のnext_state内のこの構造の総表面積


    # Function to calculate reward for each agent / 各エージェントの報酬を計算する機能
    def _game_get_reward(self,agent):
        self.reward[agent] += 1000*(self.strain_e[0]-self.next_strain_e[0])/(self.int_strain_e[0]) # Reward rule / 報酬規定

        if self.game_step == self.end_step: # Check if game is end / ゲームが終了したかどうかを確認する
            self.done_counter = 1
        return self.reward[agent],self.done_counter

    # Function to reset every values and prepare for the next game / すべての値をリセットして次のゲームに備える機能
    def reset(self):
        self.state = [] # Game state / ゲームの状態
        self.action = [] # Game action / ゲームの行動
        self.reward = [] # Game reward for each agent / 各エージェントのゲーム報酬
        for i in range(self.num_agents):
            self.reward.append(0)
        self.next_state = [] # Game next state / ゲームの次の状態
        self.done = [] # Game over counter / ゲームオーバーカウンター
        self.doing = [] # List of position(x,z) in the structure of each agent / 各エージェントの構造内のposition（x、z）のリスト
        for i in range(self.num_agents): # Initialize starting position of each structure / 各構造の開始位置を初期化
            self.doing.append([0,0])
        self.game_step = 1 # Game initial step / ゲームの最初のステップ
        self.xmax = 0 # Maximum x coordinate value in this structural model (horizontal) / この構造モデルの最大x座標値（水平）
        self.xmin = 0 # Minimum x coordinate value in this structural model (horizontal) / この構造モデル（水平）の最小x座標値
        self.ymax = 0 # Maximum y coordinate value in this structural model (vertical) / この構造モデルのY座標の最大値（垂直）
        self.ymin = 0 # Minimum y coordinate value in this structural model (vertical) / この構造モデルのY座標の最小値（垂直）
        self.zmax = 0 # Maximum z coordinate value in this structural model (horizontal) / この構造モデルの最大Z座標値（水平）
        self.zmin = 0 # Minimum z coordinate value in this structural model (horizontal) / この構造モデルの最小Z座標値（水平）
        self.sval = 0.001 # small noise / 小さなノイズ

        self.dymax = 0
        self.dymin = 0
        self.sdyval = 0
        self.dmax0 = 0
        self.dmaxset = 0
        self.dtxmax = 0
        self.dtxmin = 0
        self.dtymax = 0
        self.dtymin = 0
        self.dtzmax = 0
        self.dtzmin = 0
        self.dtxmax0 = 0
        self.dtymax0 = 0
        self.dtzmax0 = 0

        #**********
        self.int_strain_e = 0 # Initial Total length for this game / このゲームの初期の全長
        self.strain_e = 0 # Current Total length  for this game / このゲームの現在の合計の長さ
        self.next_strain_e = 0 # Total length  after agents do actions. Used for calculating reward / エージェントがアクションを実行した後の全長。 報酬の計算に使用されます
        #**********
        self.reward_counter = [] # List of reward of each agent / 各エージェントの報酬一覧
        for i in range(self.num_agents): # Initialize reward for each agent / 各エージェントの報酬を初期化する
            self.reward.append(0)
            self.reward_counter.append(0)
        self.done_counter = 0 # Counter for game end / ゲーム終了のカウンター

    # Function change state to next_state / 関数は状態を次の状態に変更します
    def step(self):
        self.state = self.next_state
        self.action = [] # Reset List of Action for each agent / 各エージェントのアクションリストをリセット
        for i in range(len(self.reward)): # Reset List of Reward for each agent / 各エージェントの報酬リストをリセット
            self.reward[i] = 0
        self.next_state = [] # Reset List of next state for each agent / 各エージェントの次の状態のリストをリセット
        self.done = [] # Reset List of game over counter / ゲームオーバーカウンターのリストをリセット
        self.game_step += 1 # Increase game step counter / ゲームのステップカウンターを増やす


#=============================================================================
# GAME 2 '研究室'
class Game2:
    def __init__(self,end_step,alpha,max_y_val,model,num_agents=1,render=0,tell_action=False):
        self.name = 'GAME 2' # Name of the game / ゲームの名前
        self.description = 'AGENT HAS 6 ACTIONS:  MOVE NODE (UP DOWN), MOVE TO SURROUNDING NODES (LEFT RIGHT UP DOWN)' # Game's description / ゲームの説明
        self.objective = 'REDUCE STRAIN ENERGY' # Game's objective / ゲームの目的
        self.tell_action =tell_action # Print agent action in console /コンソールでエージェントの行動を印刷する
        self.num_agents = num_agents # Amount of agents in the game / ゲーム内のエージェントの数
        self.gen_model = model # Gen structural model used in the game / ゲームで使用されるGen構造モデル
        self.model = model.model # Structural model used in the game / ゲームで使用される構造モデル
        self.num_x = model.num_x # Amount of Structural model's span in x axis (horizontal) / X軸での構造モデルのスパンの量（水平）
        self.num_z = model.num_z # Amount of Structural model's span in z axis (horizontal) / Z軸での構造モデルのスパンの量（水平）
        self.render = render # Render after each step / 各ステップ後にレンダリング
        self.game_step = 1 # Game initial step / ゲームの最初のステップ
        self.game_type = 0 # Game's state type / ゲームの状態タイプ
        self.end_step = end_step # Game final step / ゲームの最終ステップ
        self.alpha = alpha # Magnitude for agents to adjust structure as a factor of Structural model's span / エージェントが構造モデルのスパンの要素として構造を調整するための大きさ
        self.y_step = self.alpha*self.gen_model.span # Magnitude for agents to adjust structure(m) / エージェントが構造を調整するための大きさ（m）

        self.state = [] # Game state / ゲームの状態
        self.action = [] # Game action / ゲームの行動
        self.reward = [] # Game reward for each agent / 各エージェントのゲーム報酬
        self.next_state = [] # Game next state / ゲームの次の状態
        self.done = [] # Game over counter / ゲームオーバーカウンター

        self.doing = [] # List of position(x,z) in the structure of each agent / 各エージェントの構造内のposition（x、z）のリスト
        for i in range(self.num_agents): # Initialize starting position of each structure / 各構造の開始位置を初期化
            self.doing.append([0,0])
        self.metagrid = [] # 2-D Array of data in each structural node / 各構造ノードのデータの2次元配列
        for i in range(self.num_z): # Initialize structural data array / 構造データ配列を初期化する
            self.metagrid.append([])
            for j in range(self.num_x):
                self.metagrid[-1].append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) # Maximum num_state in this suite is 23 / このスイートの最大num_stateは23です
        self.xmax = 0 # Maximum x coordinate value in this structural model (horizontal) / この構造モデルの最大x座標値（水平）
        self.xmin = 0 # Minimum x coordinate value in this structural model (horizontal) / この構造モデル（水平）の最小x座標値
        self.ymax = 0 # Maximum y coordinate value in this structural model (vertical) / この構造モデルのY座標の最大値（垂直）
        self.ymin = 0 # Minimum y coordinate value in this structural model (vertical) / この構造モデルのY座標の最小値（垂直）
        self.zmax = 0 # Maximum z coordinate value in this structural model (horizontal) / この構造モデルの最大Z座標値（水平）
        self.zmin = 0 # Minimum z coordinate value in this structural model (horizontal) / この構造モデルの最小Z座標値（水平）
        self.sval = 0.001 # small noise / 小さなノイズ

        self.dymax = 0
        self.dymin = 0
        self.sdyval = 0
        self.dmax0 = 0
        self.dmaxset = 0
        self.dtxmax = 0
        self.dtxmin = 0
        self.dtymax = 0
        self.dtymin = 0
        self.dtzmax = 0
        self.dtzmin = 0
        self.dtxmax0 = 0
        self.dtymax0 = 0
        self.dtzmax0 = 0

        self.max_y_val = max_y_val # Maximum y coordinate value in this structural model for this game (vertical) / このゲームのこの構造モデルの最大y座標値（垂直）
        #**********
        self.int_strain_e = 0 # Initial strain energy for this game / このゲームの初期ひずみエネルギー
        self.strain_e = 0 # Current strain energy for this game / このゲームの現在のひずみエネルギー
        self.next_strain_e = 0 # Strain energy after agents do actions. Used for calculating reward / エージェントがアクションを実行した後のエネルギーのひずみ。 報酬の計算に使用されます
        #**********
        self.reward_counter = [] # List of reward of each agent / 各エージェントの報酬一覧
        for i in range(self.num_agents): # Initialize reward for each agent / 各エージェントの報酬を初期化する
            self.reward.append(0)
            self.reward_counter.append(0)
        self.done_counter = 0 # Counter for game end / ゲーム終了のカウンター

        # Print out game properties at beginning of the game / ゲームの開始時にゲームのプロパティを印刷する
        print('-------------------------------------------------------')
        print(self.description)
        print(self.objective)
        print('GAME WILL BE ENDED AFTER {} STEP'.format(self.end_step))
        print('-------------------------------------------------------')

    # Function to set Game's state type / ゲームの状態タイプを設定する関数
    def set_gametype(self,game_type):
        self.game_type = game_type

    # Fuction to update structural data array / 構造化データ配列を更新する関数
    def _update_metagrid(self):
        # update structure
        self.model.restore()
        self.model.gen_all()

        xlist = []
        ylist = []
        zlist = []
        dylist = []
        dtxlist = []
        dtylist = []
        dtzlist = []

        for i in range(len(self.model.nodes)):
            xlist.append(self.model.nodes[i].coord[0])
            ylist.append(self.model.nodes[i].coord[1])
            zlist.append(self.model.nodes[i].coord[2])
            dylist.append(self.model.nodes[i].global_d[1][0])
            dtxlist.append(self.model.nodes[i].global_d[3][0])
            dtylist.append(self.model.nodes[i].global_d[4][0])
            dtzlist.append(self.model.nodes[i].global_d[5][0])

        self.xmax = max(xlist)
        self.xmin = min(xlist)
        self.ymax = max(ylist)
        self.ymin = min(ylist)
        self.zmax = max(zlist)
        self.zmin = min(zlist)
        self.sdyval = self.sval*self.dymin
        self.dtxmax = max(dtxlist)
        self.dtxmin = min(dtxlist)
        self.dtymax = max(dtylist)
        self.dtymin = min(dtylist)
        self.dtzmax = max(dtzlist)
        self.dtzmin = min(dtzlist)
        self.dymax = max(dylist)
        self.dymin = min(dylist)

        if self.dmaxset == 0:
            self.dmax0 = abs(min(dylist))
            self.dtxmax0 = max([abs(min(dtxlist)),abs(max(dtxlist))])
            self.dtymax0 = max([abs(min(dtylist)),abs(max(dtylist))])
            self.dtzmax0 = max([abs(min(dtzlist)),abs(max(dtzlist))])

            self.dmaxset = 1

        for i in range(self.num_z):
            for j in range(self.num_x):
                dtxi,dtzi,dtxj_up,dtxj_down,dtxj_left,dtxj_right,dtzj_up,dtzj_down,dtzj_left,dtzj_right,di,dj_up,dj_down,dj_left,dj_right,n_up,n_down,n_left,n_right,pos1,pos2,bc,geo = state_data(i,j,self)

                self.metagrid[i][j] = [[dtxi],[dtzi],[dtxj_up],[dtxj_down],[dtxj_left],[dtxj_right],[dtzj_up],[dtzj_down],[dtzj_left],[dtzj_right],[di],[dj_up],[dj_down],[dj_left],[dj_right],[n_up],[n_down],[n_left],[n_right],[pos1],[pos2],[bc],[geo]]

                '''
                self.metagrid[i][j] = [
                [n_up],[n_down],[n_left],[n_right],
                [pos1],[pos2],
                [geo]
                ]
                '''

    # Function to calculate value use in the reward system / 報酬制度での価値利用を計算する機能
    def _game_gen_state_condi(self):
        self.model.restore() # Reset structural model's values / 構造モデルの値をリセットする
        self.model.gen_all() # Calculate total length / 全長を計算する
        self.strain_e = self.model.U_full # Current total length of this structure / この構造の現在の全長
        if self.game_step == 1: # Initial total length of this structure / この構造の初期の全長
            self.int_strain_e = self.model.U_full
        else:
            pass

    # Function to initialize state / 状態を初期化する関数
    def _game_get_1_state(self,do):
        self._update_metagrid() # update structural data array / 構造データ配列を更新する
        # do = [i,j]
        # metagrid[z,x]
        # Check game type to generate state from structural data array / 構造データ配列から状態を生成するゲームタイプをチェックしてください

        x = self.metagrid[do[0]][do[1]]
        state = np.array(x)# state
        return state

    # Function to generate next state / 次の状態を生成する関数
    def _game_get_next_state(self,do,action,i=0):
        num = [action[0][0],action[0][1],action[0][2],action[0][3],action[0][4],action[0][5]]
        num = num.index(max(num)) # Find maximum index of  the action receive from Neural Network / ニューラルネットワークから受け取るアクションの最大インデックスを見つける
        # next_state = f(action)
        # Interpretation of action index / 行動のインデックスの解釈
        if num == 0: # Adjust this node by moving up in the magnitude of y_step / y_stepの大きさを上に移動して、このノードを調整します
            if self.gen_model.n_u_name_div[do[0]][do[1]].res[1] !=1:
                if self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] + self.y_step <= self.max_y_val:
                    self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] += self.y_step
                else:
                    pass
            else:
                pass
        elif num == 1: # Adjust this node by moving down in the magnitude of y_step / y_stepの大きさを下に移動して、このノードを調整します
            if self.gen_model.n_u_name_div[do[0]][do[1]].res[1] != 1:
                if self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] - self.y_step >= 0:
                    self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] -= self.y_step
                else:
                    pass
            else:
                pass
        elif num == 2: # Agent move to other node to the right(move right x+1) / ＃エージェントは他のノードに右に移動します（右にx + 1移動）
            # do[z,x]
            if (do[1]+1 != (len(self.gen_model.n_u_name_div[0]))):
                self.doing[0][1] = do[1]+1
            else:
                pass
        elif num == 3: # Agent move to other node to the left(move left x-1) / エージェントは他のノードに左に移動します（左に移動x-1）
            # do[z,x]
            if (do[1] != 0):
                self.doing[0][1] = do[1]-1
            else:
                pass
        elif num == 4: # Agent move to other node to the upper(move up z-1) / エージェントが他のノードに移動します（上に移動z-1）
            # do[z,x]
            if (do[0] != 0):
                self.doing[0][0] = do[0]-1
            else:
                pass
        elif num == 5: # Agent move to other node to the lower(move down z+1) / エージェントは他のノードに移動します（z + 1に移動）
            # do[z,x]
            if (do[0]+1 != (len(self.gen_model.n_u_name_div))):
                self.doing[0][0] = do[0]+1
            else:
                pass

        announce = ['z_up','z_down','move right','move left','move up','move down'] # list of actions / 行動のリスト
        if self.tell_action == True:
            print(announce[num-1]) # print out action if tell_action is Trues / tell_actionがTrueの場合、行動を出力します
        self._update_metagrid()  # update structural data array / 構造データ配列を更新する
        # Check game type to generate state from structural data array / 構造データ配列から状態を生成するゲームタイプをチェックしてください

        x = self.metagrid[self.doing[i][0]][self.doing[i][1]]
        next_state = np.array(x) # next_state / 次の状態
        return next_state

    # Function to calculate value use in the reward system / 報酬制度での価値利用を計算する機能
    def _gen_gen_reward_condition(self):
        # Calculate next state total length / 次の状態の全長を計算する
        self.model.restore() # Reset structural model's values / 構造モデルの値をリセットする
        self.model.gen_all() # Calculate Strain energy / ひずみエネルギーを計算する
        self.next_strain_e = self.model.U_full # Strain energy of this structure in the next_state after agents do actions / エージェントがアクションを実行した後の次の状態におけるこの構造のひずみエネルギー

    # Function to calculate reward for each agent / 各エージェントの報酬を計算する機能
    def _game_get_reward(self,agent):
        self.reward[agent] += 1000*(self.strain_e[0]-self.next_strain_e[0])/(self.int_strain_e[0]) # Reward rule / 報酬規定
        if self.game_step == self.end_step: # Check if game is end / ゲームが終了したかどうかを確認する
            self.done_counter = 1
        return self.reward[agent],self.done_counter

    # Function to reset every values and prepare for the next game / すべての値をリセットして次のゲームに備える機能
    def reset(self):
        self.state = [] # Game state / ゲームの状態
        self.action = [] # Game action / ゲームの行動
        self.reward = [] # Game reward for each agent / 各エージェントのゲーム報酬
        for i in range(self.num_agents):
            self.reward.append(0)
        self.next_state = [] # Game next state / ゲームの次の状態
        self.done = [] # Game over counter / ゲームオーバーカウンター
        self.doing = [] # List of position(x,z) in the structure of each agent / 各エージェントの構造内のposition（x、z）のリスト
        for i in range(self.num_agents): # Initialize starting position of each structure / 各構造の開始位置を初期化
            self.doing.append([0,0])
        self.game_step = 1 # Game initial step / ゲームの最初のステップ
        self.xmax = 0 # Maximum x coordinate value in this structural model (horizontal) / この構造モデルの最大x座標値（水平）
        self.xmin = 0 # Minimum x coordinate value in this structural model (horizontal) / この構造モデル（水平）の最小x座標値
        self.ymax = 0 # Maximum y coordinate value in this structural model (vertical) / この構造モデルのY座標の最大値（垂直）
        self.ymin = 0 # Minimum y coordinate value in this structural model (vertical) / この構造モデルのY座標の最小値（垂直）
        self.zmax = 0 # Maximum z coordinate value in this structural model (horizontal) / この構造モデルの最大Z座標値（水平）
        self.zmin = 0 # Minimum z coordinate value in this structural model (horizontal) / この構造モデルの最小Z座標値（水平）
        self.sval = 0.001 # small noise / 小さなノイズ

        self.dymax = 0
        self.dymin = 0
        self.sdyval = 0
        self.dmax0 = 0
        self.dmaxset = 0
        self.dtxmax = 0
        self.dtxmin = 0
        self.dtymax = 0
        self.dtymin = 0
        self.dtzmax = 0
        self.dtzmin = 0
        self.dtxmax0 = 0
        self.dtymax0 = 0
        self.dtzmax0 = 0

        #**********
        self.int_strain_e = 0 # Initial strain energy for this game / このゲームの初期ひずみエネルギー
        self.strain_e = 0 # Current strain energy for this game / このゲームの現在のひずみエネルギー
        self.next_strain_e = 0 # Strain energy after agents do actions. Used for calculating reward / エージェントがアクションを実行した後のエネルギーのひずみ。 報酬の計算に使用されます
        #**********
        self.reward_counter = [] # List of reward of each agent / 各エージェントの報酬一覧
        for i in range(self.num_agents): # Initialize reward for each agent / 各エージェントの報酬を初期化する
            self.reward.append(0)
            self.reward_counter.append(0)
        self.done_counter = 0 # Counter for game end / ゲーム終了のカウンター

    # Function change state to next_state / 関数は状態を次の状態に変更します
    def step(self):
        self.state = self.next_state
        self.action = [] # Reset List of Action for each agent / 各エージェントのアクションリストをリセット
        for i in range(len(self.reward)): # Reset List of Reward for each agent / 各エージェントの報酬リストをリセット
            self.reward[i] = 0
        self.next_state = [] # Reset List of next state for each agent / 各エージェントの次の状態のリストをリセット
        self.done = [] # Reset List of game over counter / ゲームオーバーカウンターのリストをリセット
        self.game_step += 1 # Increase game step counter / ゲームのステップカウンターを増やす

#=============================================================================
# GAME 3
class Game3:
    def __init__(self,end_step,alpha,max_y_val,model,num_agents=1,render=0,tell_action=False):
        self.name = 'GAME 3'
        self.description = 'AGENT HAS 2 SUB ACTIONS:  MOVE NODE (UP DOWN), MOVE TO SURROUNDING NODES (LEFT RIGHT UP DOWN)'
        self.objective = 'REDUCE TOTAL SURFACE'
        self.tell_action =tell_action
        self.num_agents = num_agents
        self.gen_model = model
        self.model = model.model
        self.num_x = model.num_x
        self.num_z = model.num_z
        self.render = render # if render==0 no render
        self.game_step = 1
        self.game_type = 0
        self.end_step = end_step # when will the game end
        self.alpha = alpha # magnitude of adjusting node as a factor of span
        self.y_step = self.alpha*self.gen_model.span

        #=======================
        #State Action'
        # for MADDPG
        #=======================
        self.state = []
        self.action = [] #get action from rl or data file
        self.reward = []
        self.next_state = []
        self.done = []

        #=======================
        # Game rules
        #=======================
        self.doing = [] # list of doing node of each agent
        self.doing = [[1,1],[len(self.gen_model.n_u_name_div)-2,len(self.gen_model.n_u_name_div[0])-2],
                     [1,len(self.gen_model.n_u_name_div[0])-2],
                     [len(self.gen_model.n_u_name_div)-2,1]
                     ]

        self.metagrid = []
        for i in range(self.num_z):
            self.metagrid.append([])
            for j in range(self.num_x):
                self.metagrid[-1].append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) # Maximum num_state in this suite is 23 / このスイートの最大num_stateは23です

        self.xmax = 0
        self.xmin = 0
        self.ymax = 0
        self.ymin = 0
        self.zmax = 0
        self.zmin = 0
        self.sval = 0.001 # small noise

        # =========================================
        # deformation
        # =========================================

        self.dymax = 0
        self.dymin = 0
        self.sdyval = 0
        self.dmax0 = 0
        self.dmaxset = 0
        self.dtxmax = 0
        self.dtxmin = 0
        self.dtymax = 0
        self.dtymin = 0
        self.dtzmax = 0
        self.dtzmin = 0
        self.dtxmax0 = 0
        self.dtymax0 = 0
        self.dtzmax0 = 0

        self.max_y_val = max_y_val
        self.bound = [self.y_step,self.y_step,1,1,1,1] #Adjusting has bound of self.max_y_val,djusting has bound of self.max_y_val, Moving is 1 or 0
        self.int_strain_e = 0
        self.strain_e = 0
        self.next_strain_e =0
        self.reward_counter = []
        for i in range(self.num_agents):
            self.reward.append(0)
            self.reward_counter.append(0)
        self.done_counter = 0

        print('-------------------------------------------------------')
        print(self.description)
        print(self.objective)
        print('GAME WILL BE ENDED AFTER {} STEP'.format(self.end_step))
        print('-------------------------------------------------------')

    def set_gametype(self,game_type):
        self.game_type = game_type

    def _update_metagrid(self):
        # update structure
        self.model.restore()
        self.model.gen_all()

        xlist = []
        ylist = []
        zlist = []
        dylist = []
        dtxlist = []
        dtylist = []
        dtzlist = []

        for i in range(len(self.model.nodes)):
            xlist.append(self.model.nodes[i].coord[0])
            ylist.append(self.model.nodes[i].coord[1])
            zlist.append(self.model.nodes[i].coord[2])
            dylist.append(self.model.nodes[i].global_d[1][0])
            dtxlist.append(self.model.nodes[i].global_d[3][0])
            dtylist.append(self.model.nodes[i].global_d[4][0])
            dtzlist.append(self.model.nodes[i].global_d[5][0])

        self.xmax = max(xlist)
        self.xmin = min(xlist)
        self.ymax = max(ylist)
        self.ymin = min(ylist)
        self.zmax = max(zlist)
        self.zmin = min(zlist)
        self.sdyval = self.sval*self.dymin
        self.dtxmax = max(dtxlist)
        self.dtxmin = min(dtxlist)
        self.dtymax = max(dtylist)
        self.dtymin = min(dtylist)
        self.dtzmax = max(dtzlist)
        self.dtzmin = min(dtzlist)
        self.dymax = max(dylist)
        self.dymin = min(dylist)

        if self.dmaxset == 0:
            self.dmax0 = abs(min(dylist))
            self.dtxmax0 = max([abs(min(dtxlist)),abs(max(dtxlist))])
            self.dtymax0 = max([abs(min(dtylist)),abs(max(dtylist))])
            self.dtzmax0 = max([abs(min(dtzlist)),abs(max(dtzlist))])

            self.dmaxset = 1

        for i in range(self.num_z):
            for j in range(self.num_x):
                dtxi,dtzi,dtxj_up,dtxj_down,dtxj_left,dtxj_right,dtzj_up,dtzj_down,dtzj_left,dtzj_right,di,dj_up,dj_down,dj_left,dj_right,n_up,n_down,n_left,n_right,pos1,pos2,bc,geo = state_data(i,j,self)
                '''
                self.metagrid[i][j] = [[dtxi],[dtzi],[dtxj_up],[dtxj_down],[dtxj_left],[dtxj_right],[dtzj_up],[dtzj_down],[dtzj_left],[dtzj_right],[di],[dj_up],[dj_down],[dj_left],[dj_right],[n_up],[n_down],[n_left],[n_right],[pos1],[pos2],[bc],[geo]]
                '''
                self.metagrid[i][j] = [
                [n_up],[n_down],[n_left],[n_right],
                [pos1],[pos2],
                [geo]
                ]

    # Function to calculate value use in the reward system / 報酬制度での価値利用を計算する機能
    def _game_gen_state_condi(self):
        self.gen_model.gen_surface1() # Calculate total surface / 総表面積を計算する
        self.strain_e = self.gen_model.surface_1 # Current total surface of this structure / この構造の総表面積
        if self.game_step == 1: # Initial total length of this structure / この構造の初期の全長
            self.int_strain_e = self.gen_model.surface_1
        else:
            pass


    def _game_get_1_state(self,do,multi=False):
        if multi:
            pass
        else:
            self._update_metagrid()
        # do = [i,j]
        # metagrid[z,x]
        '''
        if self.game_type==0:
            print('There is no game type')
        elif  self.game_type==1:
            # Theta-ij Pos Bc(numstate = 7)
            x = self.metagrid[do[0]][do[1]][5:-1]
        elif  self.game_type==2:
            # Z/Zmax Pos Bc  (numstate = 4)
            x = self.metagrid[do[0]][do[1]][9:]
        elif  self.game_type==3:
            # Theta-ij di dj  (numstate = 9)
            x = self.metagrid[do[0]][do[1]][:9]
        elif  self.game_type==4:
            # Theta-ij di dj Z/Zmax Pos Bc  (numstate = 13)
            x = self.metagrid[do[0]][do[1]]
        '''
        x = self.metagrid[do[0]][do[1]]

        state = np.array(x)
        return state

    def _game_get_next_state(self,do,action,i=0,multi=False):

        num = [action[0][0],action[0][1],action[0][2],action[0][3],action[0][4],action[0][5]]
        adjnum = [action[0][0],action[0][1]]
        movenum = [action[0][2],action[0][3],action[0][4],action[0][5]]
        movenum = movenum.index(max(movenum))
        adjnum = adjnum.index(max(adjnum))
        num = num.index(max(num))
        # next_state = f(action)
        # Interprete action
        if adjnum == 0:
            step = action[0][0]*self.bound[0]
            if self.gen_model.n_u_name_div[do[0]][do[1]].res[1] !=1:
                if self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] + step <= self.max_y_val:
                    self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] += step
                    if self.tell_action == True:
                        print('Z+:{}'.format(action[0][0]*self.bound[0]))
                else:
                    pass
            else:
                pass
                #self.bad = 1

        elif adjnum == 1:
            step = action[0][1]*self.bound[1]
            if self.gen_model.n_u_name_div[do[0]][do[1]].res[1] != 1:
                if self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] - step >= 0:
                    self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] -= step
                    if self.tell_action == True:
                        print('Z-:{}'.format(action[0][1]*self.bound[1]))
                else:
                    pass
            else:
                pass
                #self.bad = 1
        if movenum == 0:
            # move right x+1
            # do[z,x]
            if (do[1]+1 != (len(self.gen_model.n_u_name_div[0]))):
                self.doing[i][1] = do[1]+1
            else:
                #self.doing[i][1] = do[1]-1
                pass
        elif movenum == 1:
            # move left x-1
            # do[z,x]
            if (do[1] != 0):
                self.doing[i][1] = do[1]-1
            else:
                #self.doing[i][1] = do[1]+1
                pass
        elif movenum == 2:
            # move up z-1
            # do[z,x]
            if (do[0] != 0):
                self.doing[i][0] = do[0]-1
            else:
                #self.doing[i][0] = do[0]+1
                pass
        elif movenum == 3:
            # move down z+1
            # do[z,x]
            if (do[0]+1 != (len(self.gen_model.n_u_name_div))):
                self.doing[i][0] = do[0]+1
                '''
                if self.gen_model.n_u_name_div[do[0]+1][do[1]].res[1] !=1:
                    self.doing[i][0] = do[0]+1
                else:
                    pass
                '''
            else:
                #self.doing[i][0] = do[0]-1
                pass

        #announce1 = ['z_up','z_down']
        announce2 = ['move right','move left','move up','move down']
        if self.tell_action == True:
            #print(announce1[adjnum-1]) # print out action
            print(announce2[movenum-1]) # print out action

        if multi:
            pass
        else:
            self._update_metagrid()

            if self.game_type==0:
                print('There is no game type')
            elif  self.game_type==1:
                # Theta-ij Pos Bc(numstate = 7)
                x = self.metagrid[self.doing[i][0]][self.doing[i][1]][5:-1]
            elif  self.game_type==2:
                # Z/Zmax Pos Bc  (numstate = 4)
                x = self.metagrid[self.doing[i][0]][self.doing[i][1]][9:]
            elif  self.game_type==3:
                # Theta-ij di dj  (numstate = 9)
                x = self.metagrid[self.doing[i][0]][self.doing[i][1]][:9]
            elif  self.game_type==4:
                # Theta-ij di dj Z/Zmax Pos Bc  (numstate = 13)
                x = self.metagrid[self.doing[i][0]][self.doing[i][1]]

            next_state = np.array(x)
            return next_state

    def _game_get_next_state_maddpg(self,do,i):
        '''
        if self.game_type==0:
            print('There is no game type')
        elif  self.game_type==1:
            # Theta-ij Pos Bc(numstate = 7)
            x = self.metagrid[self.doing[i][0]][self.doing[i][1]][5:-1]
        elif  self.game_type==2:
            # Z/Zmax Pos Bc  (numstate = 4)
            x = self.metagrid[self.doing[i][0]][self.doing[i][1]][9:]
        elif  self.game_type==3:
            # Theta-ij di dj  (numstate = 9)
            x = self.metagrid[self.doing[i][0]][self.doing[i][1]][:9]
        elif  self.game_type==4:
            # Theta-ij di dj Z/Zmax Pos Bc  (numstate = 13)
            x = self.metagrid[self.doing[i][0]][self.doing[i][1]]
        '''
        x = self.metagrid[self.doing[i][0]][self.doing[i][1]]

        next_state = np.array(x)
        return next_state

    # Function to calculate value use in the reward system / 報酬制度での価値利用を計算する機能
    def _gen_gen_reward_condition(self):

        self.gen_model.gen_surface1() # Calculate next state's total surface / 次の状態の総表面積を計算する
        self.next_strain_e = self.gen_model.surface_1# Total surface of this structure in the next_state after agents do actions / エージェントが行動を実行した後のnext_state内のこの構造の総表面積


    # Function to calculate reward for each agent / 各エージェントの報酬を計算する機能
    def _game_get_reward(self,agent):
        self.reward[agent] += 1000*(self.strain_e[0]-self.next_strain_e[0])/(self.int_strain_e[0]) # Reward rule / 報酬規定

        if self.game_step == self.end_step: # Check if game is end / ゲームが終了したかどうかを確認する
            self.done_counter = 1
        return self.reward[agent],self.done_counter

    def reset(self):
        self.state = []
        self.action = []
        self.reward = []
        self.doing = []
        self.doing = [[1,1],[len(self.gen_model.n_u_name_div)-2,len(self.gen_model.n_u_name_div[0])-2],
                     [1,len(self.gen_model.n_u_name_div[0])-2],
                     [len(self.gen_model.n_u_name_div)-2,1]
                     ]
        for i in range(self.num_agents):
            self.reward.append(0)

        self.next_state = []
        self.done = []
        self.game_step = 1

        self.xmax = 0
        self.xmin = 0
        self.ymax = 0
        self.ymin = 0
        self.zmax = 0
        self.zmin = 0

        self.dymax = 0
        self.dymin = 0
        self.sdyval = 0
        self.dmax0 = 0
        self.dmaxset = 0
        self.dtxmax = 0
        self.dtxmin = 0
        self.dtymax = 0
        self.dtymin = 0
        self.dtzmax = 0
        self.dtzmin = 0
        self.dtxmax0 = 0
        self.dtymax0 = 0
        self.dtzmax0 = 0

        self.int_strain_e = 0
        self.strain_e = 0
        self.next_strain_e =0

        self.reward_counter = []
        for i in range(self.num_agents):
            self.reward.append(0)
            self.reward_counter.append(0)
        self.done_counter = 0


    def step(self):
        self.state = self.next_state
        self.action = []
        #self.reward = []
        for i in range(len(self.reward)):
            self.reward[i] = 0
        self.next_state = []
        self.done = []

        self.game_step += 1


#=============================================================================
# GAME 4
class Game4:
    def __init__(self,end_step,alpha,max_y_val,model,num_agents=1,render=0,tell_action=False):
        self.name = 'GAME 4'
        self.description = 'AGENT HAS 2 SUB-ACTIONS IN ONE STEP:  1. MOVE NODE (UP DOWN) IN CONTIUOUS SPACE, 2.MOVE TO SURROUNDING NODES (LEFT RIGHT UP DOWN)'
        self.objective = 'REDUCE STRAIN ENERGY'
        self.tell_action =tell_action
        self.num_agents = num_agents
        self.gen_model = model
        self.model = model.model
        self.num_x = model.num_x
        self.num_z = model.num_z
        self.render = render # if render==0 no render
        self.game_step = 1
        self.game_type = 0
        self.end_step = end_step # when will the game end
        self.alpha = alpha # magnitude of adjusting node as a factor of span
        self.y_step = self.alpha*self.gen_model.span

        #=======================
        #State Action'
        # for MADDPG
        #=======================
        self.state = []
        self.action = [] #get action from rl or data file
        self.reward = []
        self.next_state = []
        self.done = []

        #=======================
        # Game rules
        #=======================
        self.bad = 0
        self.doing = [] # list of doing node of each agent
        self.doing = [[1,1],[len(self.gen_model.n_u_name_div)-2,len(self.gen_model.n_u_name_div[0])-2],
                     [1,len(self.gen_model.n_u_name_div[0])-2],
                     [len(self.gen_model.n_u_name_div)-2,1]
                     ]
        '''
        for i in range(self.num_agents):
            self.doing.append([0,0])
        '''
        #do1:(len(self.gen_model.n_u_name_div[0])-1)
        #do0:(len(self.gen_model.n_u_name_div)-1)
        self.metagrid = []
        for i in range(self.num_z):
            self.metagrid.append([])
            for j in range(self.num_x):
                self.metagrid[-1].append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) # Maximum num_state in this suite is 23 / このスイートの最大num_stateは23です

        self.xmax = 0
        self.xmin = 0
        self.ymax = 0
        self.ymin = 0
        self.zmax = 0
        self.zmin = 0
        self.sval = 0.001 # small noise

        # =========================================
        # deformation
        # =========================================

        self.dymax = 0
        self.dymin = 0
        self.sdyval = 0
        self.dmax0 = 0
        self.dmaxset = 0
        self.dtxmax = 0
        self.dtxmin = 0
        self.dtymax = 0
        self.dtymin = 0
        self.dtzmax = 0
        self.dtzmin = 0
        self.dtxmax0 = 0
        self.dtymax0 = 0
        self.dtzmax0 = 0

        self.max_y_val = max_y_val
        self.bound = [self.y_step,self.y_step,1,1,1,1] #Adjusting has bound of self.max_y_val,djusting has bound of self.max_y_val, Moving is 1 or 0
        self.int_strain_e = 0
        self.strain_e = 0
        self.next_strain_e =0
        self.reward_counter = []
        for i in range(self.num_agents):
            self.reward.append(0)
            self.reward_counter.append(0)
        self.done_counter = 0

        print('-------------------------------------------------------')
        print(self.description)
        print(self.objective)
        print('GAME WILL BE ENDED AFTER {} STEP'.format(self.end_step))
        print('-------------------------------------------------------')

    def set_gametype(self,game_type):
        self.game_type = game_type

    def _update_metagrid(self):
        # update structure
        self.model.restore()
        self.model.gen_all()

        xlist = []
        ylist = []
        zlist = []
        dylist = []
        dtxlist = []
        dtylist = []
        dtzlist = []

        for i in range(len(self.model.nodes)):
            xlist.append(self.model.nodes[i].coord[0])
            ylist.append(self.model.nodes[i].coord[1])
            zlist.append(self.model.nodes[i].coord[2])
            dylist.append(self.model.nodes[i].global_d[1][0])
            dtxlist.append(self.model.nodes[i].global_d[3][0])
            dtylist.append(self.model.nodes[i].global_d[4][0])
            dtzlist.append(self.model.nodes[i].global_d[5][0])

        self.xmax = max(xlist)
        self.xmin = min(xlist)
        self.ymax = max(ylist)
        self.ymin = min(ylist)
        self.zmax = max(zlist)
        self.zmin = min(zlist)
        self.sdyval = self.sval*self.dymin
        self.dtxmax = max(dtxlist)
        self.dtxmin = min(dtxlist)
        self.dtymax = max(dtylist)
        self.dtymin = min(dtylist)
        self.dtzmax = max(dtzlist)
        self.dtzmin = min(dtzlist)
        self.dymax = max(dylist)
        self.dymin = min(dylist)

        if self.dmaxset == 0:
            self.dmax0 = abs(min(dylist))
            self.dtxmax0 = max([abs(min(dtxlist)),abs(max(dtxlist))])
            self.dtymax0 = max([abs(min(dtylist)),abs(max(dtylist))])
            self.dtzmax0 = max([abs(min(dtzlist)),abs(max(dtzlist))])

            self.dmaxset = 1

        for i in range(self.num_z):
            for j in range(self.num_x):
                dtxi,dtzi,dtxj_up,dtxj_down,dtxj_left,dtxj_right,dtzj_up,dtzj_down,dtzj_left,dtzj_right,di,dj_up,dj_down,dj_left,dj_right,n_up,n_down,n_left,n_right,pos1,pos2,bc,geo = state_data(i,j,self)

                self.metagrid[i][j] = [[dtxi],[dtzi],[dtxj_up],[dtxj_down],[dtxj_left],[dtxj_right],[dtzj_up],[dtzj_down],[dtzj_left],[dtzj_right],[di],[dj_up],[dj_down],[dj_left],[dj_right],[n_up],[n_down],[n_left],[n_right],[pos1],[pos2],[bc],[geo]]

                '''
                self.metagrid[i][j] = [
                [n_up],[n_down],[n_left],[n_right],
                [pos1],[pos2],
                [geo]
                ]
                '''

    def _game_gen_state_condi(self):
        # Calculate strain energy
        self.model.restore()
        self.model.gen_all()
        self.strain_e = self.model.U_full
        # Calculate initial strain energy
        if self.game_step == 1:
            self.int_strain_e = self.model.U_full
        else:
            pass

    def _game_get_1_state(self,do):
        self._update_metagrid()
        # do = [i,j]
        # metagrid[z,x]
        '''
        if self.game_type==0:
            print('There is no game type')
        elif  self.game_type==1:
            # Theta-ij Pos Bc(numstate = 7)
            x = self.metagrid[do[0]][do[1]][5:-1]
        elif  self.game_type==2:
            # Z/Zmax Pos Bc  (numstate = 4)
            x = self.metagrid[do[0]][do[1]][9:]
        elif  self.game_type==3:
            # Theta-ij di dj  (numstate = 9)
            x = self.metagrid[do[0]][do[1]][:9]
        elif  self.game_type==4:
            # Theta-ij di dj Z/Zmax Pos Bc  (numstate = 13)
            x = self.metagrid[do[0]][do[1]]
        '''
        x = self.metagrid[do[0]][do[1]]
        state = np.array(x)
        return state

    def _game_get_next_state(self,do,action,i=0):

        num = [action[0][0],action[0][1],action[0][2],action[0][3],action[0][4],action[0][5]]
        adjnum = [action[0][0],action[0][1]]
        movenum = [action[0][2],action[0][3],action[0][4],action[0][5]]
        movenum = movenum.index(max(movenum))
        adjnum = adjnum.index(max(adjnum))
        num = num.index(max(num))
        # next_state = f(action)
        # Interprete action
        if adjnum == 0:
            step = action[0][0]*self.bound[0]
            if self.gen_model.n_u_name_div[do[0]][do[1]].res[1] !=1:
                if self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] + step <= self.max_y_val:
                    self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] += step
                    if self.tell_action == True:
                        print('Z+:{}'.format(action[0][0]*self.bound[0]))
                else:
                    pass
            else:
                pass
                #self.bad = 1

        elif adjnum == 1:
            step = action[0][1]*self.bound[1]
            if self.gen_model.n_u_name_div[do[0]][do[1]].res[1] != 1:
                if self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] - step >= 0:
                    self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] -= step
                    if self.tell_action == True:
                        print('Z-:{}'.format(action[0][1]*self.bound[1]))
                else:
                    pass
            else:
                pass
                #self.bad = 1
        if movenum == 0:
            # move right x+1
            # do[z,x]
            if (do[1]+1 != (len(self.gen_model.n_u_name_div[0]))):
                if self.gen_model.n_u_name_div[do[0]][do[1]+1].res[1] !=1:
                    self.doing[i][1] = do[1]+1
                else:
                    pass
            else:
                #self.doing[i][1] = do[1]-1
                pass
        elif movenum == 1:
            # move left x-1
            # do[z,x]
            if (do[1] != 0):
                if self.gen_model.n_u_name_div[do[0]][do[1]-1].res[1] !=1:
                    self.doing[i][1] = do[1]-1
                else:
                    pass
            else:
                #self.doing[i][1] = do[1]+1
                pass
        elif movenum == 2:
            # move up z-1
            # do[z,x]
            if (do[0] != 0):
                if self.gen_model.n_u_name_div[do[0]-1][do[1]].res[1] !=1:
                    self.doing[i][0] = do[0]-1
                else:
                    pass
            else:
                #self.doing[i][0] = do[0]+1
                pass
        elif movenum == 3:
            # move down z+1
            # do[z,x]
            if (do[0]+1 != (len(self.gen_model.n_u_name_div))):
                if self.gen_model.n_u_name_div[do[0]+1][do[1]].res[1] !=1:
                    self.doing[i][0] = do[0]+1
                else:
                    pass
            else:
                #self.doing[i][0] = do[0]-1
                pass

        #announce1 = ['z_up','z_down']
        announce2 = ['move right','move left','move up','move down']
        if self.tell_action == True:
            #print(announce1[adjnum-1]) # print out action
            print(announce2[movenum-1]) # print out action

        self._update_metagrid()

        x = self.metagrid[self.doing[i][0]][self.doing[i][1]]

        next_state = np.array(x)
        return next_state

    def _gen_gen_reward_condition(self):
        # Calculate next state strain energy
        self.model.restore()
        self.model.gen_all()
        self.next_strain_e = self.model.U_full

    def _game_get_reward(self,agent):

        self.reward[agent] += 1000*(self.strain_e[0]-self.next_strain_e[0])/(self.int_strain_e[0])

        if self.game_step == self.end_step:
            self.done_counter = 1

        return self.reward[agent],self.done_counter

    def reset(self):
        self.state = []
        self.action = []
        self.reward = []
        self.doing = []
        self.doing = [[1,1],[len(self.gen_model.n_u_name_div)-2,len(self.gen_model.n_u_name_div[0])-2],
                     [1,len(self.gen_model.n_u_name_div[0])-2],
                     [len(self.gen_model.n_u_name_div)-2,1]
                     ]
        for i in range(self.num_agents):
            self.reward.append(0)
            #self.doing.append([0,0])

        self.next_state = []
        self.done = []
        self.game_step = 1

        self.xmax = 0
        self.xmin = 0
        self.ymax = 0
        self.ymin = 0
        self.zmax = 0
        self.zmin = 0

        self.dymax = 0
        self.dymin = 0
        self.sdyval = 0
        self.dmax0 = 0
        self.dmaxset = 0
        self.dtxmax = 0
        self.dtxmin = 0
        self.dtymax = 0
        self.dtymin = 0
        self.dtzmax = 0
        self.dtzmin = 0
        self.dtxmax0 = 0
        self.dtymax0 = 0
        self.dtzmax0 = 0

        self.int_strain_e = 0
        self.strain_e = 0
        self.next_strain_e =0

        self.reward_counter = []
        for i in range(self.num_agents):
            self.reward.append(0)
            self.reward_counter.append(0)
        self.done_counter = 0

        self.bad = 0


    def step(self):
        self.state = self.next_state
        self.action = []
        #self.reward = []
        for i in range(len(self.reward)):
            self.reward[i] = 0
        self.next_state = []
        self.done = []

        self.game_step += 1

#=============================================================================
# GAME 5
class Game5:
    def __init__(self,end_step,alpha,max_y_val,model,num_agents=1,render=0,tell_action=False):
        self.name = 'GAME 5'
        self.description = 'AGENT HAS 2 SUB ACTIONS:  MOVE NODE (UP DOWN), MOVE TO SURROUNDING NODES (LEFT RIGHT UP DOWN)'
        self.objective = 'REDUCE TOTAL SURFACE'
        self.tell_action =tell_action
        self.num_agents = num_agents
        self.gen_model = model
        self.model = model.model
        self.num_x = model.num_x
        self.num_z = model.num_z
        self.render = render # if render==0 no render
        self.game_step = 1
        self.game_type = 0
        self.end_step = end_step # when will the game end
        self.alpha = alpha # magnitude of adjusting node as a factor of span
        self.y_step = self.alpha*self.gen_model.span

        #=======================
        #State Action'
        # for MADDPG
        #=======================
        self.state = []
        self.action = [] #get action from rl or data file
        self.reward = []
        self.next_state = []
        self.done = []

        #=======================
        # Game rules
        #=======================
        self.doing = [] # list of doing node of each agent
        '''
        self.doing = [[1,1],[len(self.gen_model.n_u_name_div)-2,len(self.gen_model.n_u_name_div[0])-2],
                     [1,len(self.gen_model.n_u_name_div[0])-2],
                     [len(self.gen_model.n_u_name_div)-2,1]
                     ]
        '''
        self.doing = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]

        self.metagrid = []
        for i in range(self.num_z):
            self.metagrid.append([])
            for j in range(self.num_x):
                self.metagrid[-1].append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) # Maximum num_state in this suite is 23 / このスイートの最大num_stateは23です

        self.xmax = 0
        self.xmin = 0
        self.ymax = 0
        self.ymin = 0
        self.zmax = 0
        self.zmin = 0
        self.sval = 0.001 # small noise

        # =========================================
        # deformation
        # =========================================

        self.dymax = 0
        self.dymin = 0
        self.sdyval = 0
        self.dmax0 = 0
        self.dmaxset = 0
        self.dtxmax = 0
        self.dtxmin = 0
        self.dtymax = 0
        self.dtymin = 0
        self.dtzmax = 0
        self.dtzmin = 0
        self.dtxmax0 = 0
        self.dtymax0 = 0
        self.dtzmax0 = 0

        self.max_y_val = max_y_val
        self.bound = [self.y_step,self.y_step,1,1,1,1] #Adjusting has bound of self.max_y_val,djusting has bound of self.max_y_val, Moving is 1 or 0
        self.int_strain_e = 0
        self.strain_e = 0
        self.next_strain_e =0
        self.reward_counter = []
        for i in range(self.num_agents):
            self.reward.append(0)
            self.reward_counter.append(0)
        self.done_counter = 0

        print('-------------------------------------------------------')
        print(self.description)
        print(self.objective)
        print('GAME WILL BE ENDED AFTER {} STEP'.format(self.end_step))
        print('-------------------------------------------------------')

    def set_gametype(self,game_type):
        self.game_type = game_type

    def _update_metagrid(self):
        # update structure
        self.model.restore()
        self.model.gen_all()

        xlist = []
        ylist = []
        zlist = []
        dylist = []
        dtxlist = []
        dtylist = []
        dtzlist = []

        for i in range(len(self.model.nodes)):
            xlist.append(self.model.nodes[i].coord[0])
            ylist.append(self.model.nodes[i].coord[1])
            zlist.append(self.model.nodes[i].coord[2])
            dylist.append(self.model.nodes[i].global_d[1][0])
            dtxlist.append(self.model.nodes[i].global_d[3][0])
            dtylist.append(self.model.nodes[i].global_d[4][0])
            dtzlist.append(self.model.nodes[i].global_d[5][0])

        self.xmax = max(xlist)
        self.xmin = min(xlist)
        self.ymax = max(ylist)
        self.ymin = min(ylist)
        self.zmax = max(zlist)
        self.zmin = min(zlist)
        self.sdyval = self.sval*self.dymin
        self.dtxmax = max(dtxlist)
        self.dtxmin = min(dtxlist)
        self.dtymax = max(dtylist)
        self.dtymin = min(dtylist)
        self.dtzmax = max(dtzlist)
        self.dtzmin = min(dtzlist)
        self.dymax = max(dylist)
        self.dymin = min(dylist)

        if self.dmaxset == 0:
            self.dmax0 = abs(min(dylist))
            self.dtxmax0 = max([abs(min(dtxlist)),abs(max(dtxlist))])
            self.dtymax0 = max([abs(min(dtylist)),abs(max(dtylist))])
            self.dtzmax0 = max([abs(min(dtzlist)),abs(max(dtzlist))])

            self.dmaxset = 1

        for i in range(self.num_z):
            for j in range(self.num_x):
                dtxi,dtzi,dtxj_up,dtxj_down,dtxj_left,dtxj_right,dtzj_up,dtzj_down,dtzj_left,dtzj_right,di,dj_up,dj_down,dj_left,dj_right,n_up,n_down,n_left,n_right,pos1,pos2,bc,geo = state_data(i,j,self)
                '''
                self.metagrid[i][j] = [[dtxi],[dtzi],[dtxj_up],[dtxj_down],[dtxj_left],[dtxj_right],[dtzj_up],[dtzj_down],[dtzj_left],[dtzj_right],[di],[dj_up],[dj_down],[dj_left],[dj_right],[n_up],[n_down],[n_left],[n_right],[pos1],[pos2],[bc],[geo]]
                '''
                self.metagrid[i][j] = [
                [n_up],[n_down],[n_left],[n_right],
                [pos1],[pos2],
                [geo]
                ]

    def _game_gen_state_condi(self):
        self.gen_model.gen_surface1() # Calculate total surface / 総表面積を計算する
        self.strain_e = self.gen_model.surface_1 # Current total surface of this structure / この構造の総表面積
        if self.game_step == 1: # Initial total length of this structure / この構造の初期の全長
            self.int_strain_e = self.gen_model.surface_1
        else:
            pass

    def _game_get_1_state(self,do,multi=False):
        if multi:
            pass
        else:
            self._update_metagrid()
        # do = [i,j]
        # metagrid[z,x]
        '''
        if self.game_type==0:
            print('There is no game type')
        elif  self.game_type==1:
            # Theta-ij Pos Bc(numstate = 7)
            x = self.metagrid[do[0]][do[1]][5:-1]
        elif  self.game_type==2:
            # Z/Zmax Pos Bc  (numstate = 4)
            x = self.metagrid[do[0]][do[1]][9:]
        elif  self.game_type==3:
            # Theta-ij di dj  (numstate = 9)
            x = self.metagrid[do[0]][do[1]][:9]
        elif  self.game_type==4:
            # Theta-ij di dj Z/Zmax Pos Bc  (numstate = 13)
            x = self.metagrid[do[0]][do[1]]
        '''
        x = self.metagrid[do[0]][do[1]]

        state = np.array(x)
        return state

    def _game_get_next_state(self,do,action,i=0,multi=False):
        num = [action[0][0],action[0][1],action[0][2],action[0][3],action[0][4],action[0][5]]
        adjnum = [action[0][0],action[0][1]]
        movenum = [action[0][2],action[0][3],action[0][4],action[0][5]]
        movenum = movenum.index(max(movenum))
        adjnum = adjnum.index(max(adjnum))
        num = num.index(max(num))
        # next_state = f(action)
        # Interprete action
        if adjnum == 0:
            step = action[0][0]*self.bound[0]
            if self.gen_model.n_u_name_div[do[0]][do[1]].res[1] !=1:
                if self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] + step <= self.max_y_val:
                    self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] += step
                    if self.tell_action == True:
                        print('Z+:{}'.format(action[0][0]*self.bound[0]))
                else:
                    pass
            else:
                pass
        elif adjnum == 1:
            step = action[0][1]*self.bound[1]
            if self.gen_model.n_u_name_div[do[0]][do[1]].res[1] != 1:
                if self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] - step >= 0:
                    self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] -= step
                    if self.tell_action == True:
                        print('Z-:{}'.format(action[0][1]*self.bound[1]))
                else:
                    pass
            else:
                pass
        if movenum == 0:
            # move right x+1
            # do[z,x]
            if (do[1]+1 != (len(self.gen_model.n_u_name_div[0]))):
                self.doing[i][1] = do[1]+1
            else:
                pass
        elif movenum == 1:
            # move left x-1
            # do[z,x]
            if (do[1] != 0):
                self.doing[i][1] = do[1]-1
            else:
                pass
        elif movenum == 2:
            # move up z-1
            # do[z,x]
            if (do[0] != 0):
                self.doing[i][0] = do[0]-1
            else:
                pass
        elif movenum == 3:
            # move down z+1
            # do[z,x]
            if (do[0]+1 != (len(self.gen_model.n_u_name_div))):
                self.doing[i][0] = do[0]+1
            else:
                pass

        #announce1 = ['z_up','z_down']
        announce2 = ['move right','move left','move up','move down']
        if self.tell_action == True:
            #print(announce1[adjnum-1]) # print out action
            print(announce2[movenum-1]) # print out action

        if multi:
            pass
        else:
            self._update_metagrid()

            if self.game_type==0:
                print('There is no game type')
            elif  self.game_type==1:
                # Theta-ij Pos Bc(numstate = 7)
                x = self.metagrid[self.doing[i][0]][self.doing[i][1]][5:-1]
            elif  self.game_type==2:
                # Z/Zmax Pos Bc  (numstate = 4)
                x = self.metagrid[self.doing[i][0]][self.doing[i][1]][9:]
            elif  self.game_type==3:
                # Theta-ij di dj  (numstate = 9)
                x = self.metagrid[self.doing[i][0]][self.doing[i][1]][:9]
            elif  self.game_type==4:
                # Theta-ij di dj Z/Zmax Pos Bc  (numstate = 13)
                x = self.metagrid[self.doing[i][0]][self.doing[i][1]]

            next_state = np.array(x)
            return next_state

    def _game_get_next_state_maddpg(self,do,i):
        '''
        if self.game_type==0:
            print('There is no game type')
        elif  self.game_type==1:
            # Theta-ij Pos Bc(numstate = 7)
            x = self.metagrid[self.doing[i][0]][self.doing[i][1]][5:-1]
        elif  self.game_type==2:
            # Z/Zmax Pos Bc  (numstate = 4)
            x = self.metagrid[self.doing[i][0]][self.doing[i][1]][9:]
        elif  self.game_type==3:
            # Theta-ij di dj  (numstate = 9)
            x = self.metagrid[self.doing[i][0]][self.doing[i][1]][:9]
        elif  self.game_type==4:
            # Theta-ij di dj Z/Zmax Pos Bc  (numstate = 13)
            x = self.metagrid[self.doing[i][0]][self.doing[i][1]]
        '''
        x = self.metagrid[self.doing[i][0]][self.doing[i][1]]
        next_state = np.array(x)
        return next_state

    # Function to calculate value use in the reward system / 報酬制度での価値利用を計算する機能
    def _gen_gen_reward_condition(self):

        self.gen_model.gen_surface1() # Calculate next state's total surface / 次の状態の総表面積を計算する
        self.next_strain_e = self.gen_model.surface_1# Total surface of this structure in the next_state after agents do actions / エージェントが行動を実行した後のnext_state内のこの構造の総表面積


    # Function to calculate reward for each agent / 各エージェントの報酬を計算する機能
    def _game_get_reward(self,agent):
        self.reward[agent] += 1000*(self.strain_e[0]-self.next_strain_e[0])/(self.int_strain_e[0]) # Reward rule / 報酬規定

        if self.game_step == self.end_step: # Check if game is end / ゲームが終了したかどうかを確認する
            self.done_counter = 1
        return self.reward[agent],self.done_counter

    def reset(self):
        self.state = []
        self.action = []
        self.reward = []
        self.doing = []
        self.doing = [[1,1],[len(self.gen_model.n_u_name_div)-2,len(self.gen_model.n_u_name_div[0])-2],
                     [1,len(self.gen_model.n_u_name_div[0])-2],
                     [len(self.gen_model.n_u_name_div)-2,1]
                     ]
        for i in range(self.num_agents):
            self.reward.append(0)

        self.next_state = []
        self.done = []
        self.game_step = 1

        self.xmax = 0
        self.xmin = 0
        self.ymax = 0
        self.ymin = 0
        self.zmax = 0
        self.zmin = 0

        self.dymax = 0
        self.dymin = 0
        self.sdyval = 0
        self.dmax0 = 0
        self.dmaxset = 0
        self.dtxmax = 0
        self.dtxmin = 0
        self.dtymax = 0
        self.dtymin = 0
        self.dtzmax = 0
        self.dtzmin = 0
        self.dtxmax0 = 0
        self.dtymax0 = 0
        self.dtzmax0 = 0

        self.int_strain_e = 0
        self.strain_e = 0
        self.next_strain_e =0

        self.reward_counter = []
        for i in range(self.num_agents):
            self.reward.append(0)
            self.reward_counter.append(0)
        self.done_counter = 0


    def step(self):
        self.state = self.next_state
        self.action = []
        #self.reward = []
        for i in range(len(self.reward)):
            self.reward[i] = 0
        self.next_state = []
        self.done = []
        self.game_step += 1



#=============================================================================
# GAME 6
class Game6:
    def __init__(self,end_step,alpha,max_y_val,model,num_agents=1,render=0,tell_action=False):
        self.name = 'GAME 6'
        self.description = 'AGENT HAS 2 SUB ACTIONS:  MOVE NODE (UP DOWN), MOVE TO SURROUNDING NODES (LEFT RIGHT UP DOWN)'
        self.objective = 'REDUCE STRAIN ENERGY'
        self.tell_action =tell_action
        self.num_agents = num_agents
        self.gen_model = model
        self.model = model.model
        self.num_x = model.num_x
        self.num_z = model.num_z
        self.render = render # if render==0 no render
        self.game_step = 1
        self.game_type = 0
        self.end_step = end_step # when will the game end
        self.alpha = alpha # magnitude of adjusting node as a factor of span
        self.y_step = self.alpha*self.gen_model.span

        #=======================
        #State Action'
        # for MADDPG
        #=======================
        self.state = []
        self.action = [] #get action from rl or data file
        self.reward = []
        self.next_state = []
        self.done = []

        #=======================
        # Game rules
        #=======================
        self.doing = [] # list of doing node of each agent
        '''
        self.doing = [[1,1],[len(self.gen_model.n_u_name_div)-2,len(self.gen_model.n_u_name_div[0])-2],
                     [1,len(self.gen_model.n_u_name_div[0])-2],
                     [len(self.gen_model.n_u_name_div)-2,1]
                     ]
        '''
        self.doing = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]

        self.metagrid = []
        for i in range(self.num_z):
            self.metagrid.append([])
            for j in range(self.num_x):
                self.metagrid[-1].append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) # Maximum num_state in this suite is 23 / このスイートの最大num_stateは23です

        self.xmax = 0
        self.xmin = 0
        self.ymax = 0
        self.ymin = 0
        self.zmax = 0
        self.zmin = 0
        self.sval = 0.001 # small noise

        # =========================================
        # deformation
        # =========================================

        self.dymax = 0
        self.dymin = 0
        self.sdyval = 0
        self.dmax0 = 0
        self.dmaxset = 0
        self.dtxmax = 0
        self.dtxmin = 0
        self.dtymax = 0
        self.dtymin = 0
        self.dtzmax = 0
        self.dtzmin = 0
        self.dtxmax0 = 0
        self.dtymax0 = 0
        self.dtzmax0 = 0

        self.max_y_val = max_y_val
        self.bound = [self.y_step,self.y_step,1,1,1,1] #Adjusting has bound of self.max_y_val,djusting has bound of self.max_y_val, Moving is 1 or 0
        self.int_strain_e = 0
        self.strain_e = 0
        self.next_strain_e =0
        self.reward_counter = []
        for i in range(self.num_agents):
            self.reward.append(0)
            self.reward_counter.append(0)
        self.done_counter = 0

        print('-------------------------------------------------------')
        print(self.description)
        print(self.objective)
        print('GAME WILL BE ENDED AFTER {} STEP'.format(self.end_step))
        print('-------------------------------------------------------')

    def set_gametype(self,game_type):
        self.game_type = game_type

    def _update_metagrid(self):
        # update structure
        self.model.restore()
        self.model.gen_all()

        xlist = []
        ylist = []
        zlist = []
        dylist = []
        dtxlist = []
        dtylist = []
        dtzlist = []

        for i in range(len(self.model.nodes)):
            xlist.append(self.model.nodes[i].coord[0])
            ylist.append(self.model.nodes[i].coord[1])
            zlist.append(self.model.nodes[i].coord[2])
            dylist.append(self.model.nodes[i].global_d[1][0])
            dtxlist.append(self.model.nodes[i].global_d[3][0])
            dtylist.append(self.model.nodes[i].global_d[4][0])
            dtzlist.append(self.model.nodes[i].global_d[5][0])

        self.xmax = max(xlist)
        self.xmin = min(xlist)
        self.ymax = max(ylist)
        self.ymin = min(ylist)
        self.zmax = max(zlist)
        self.zmin = min(zlist)
        self.sdyval = self.sval*self.dymin
        self.dtxmax = max(dtxlist)
        self.dtxmin = min(dtxlist)
        self.dtymax = max(dtylist)
        self.dtymin = min(dtylist)
        self.dtzmax = max(dtzlist)
        self.dtzmin = min(dtzlist)
        self.dymax = max(dylist)
        self.dymin = min(dylist)

        if self.dmaxset == 0:
            self.dmax0 = abs(min(dylist))
            self.dtxmax0 = max([abs(min(dtxlist)),abs(max(dtxlist))])
            self.dtymax0 = max([abs(min(dtylist)),abs(max(dtylist))])
            self.dtzmax0 = max([abs(min(dtzlist)),abs(max(dtzlist))])

            self.dmaxset = 1

        for i in range(self.num_z):
            for j in range(self.num_x):
                dtxi,dtzi,dtxj_up,dtxj_down,dtxj_left,dtxj_right,dtzj_up,dtzj_down,dtzj_left,dtzj_right,di,dj_up,dj_down,dj_left,dj_right,n_up,n_down,n_left,n_right,pos1,pos2,bc,geo = state_data(i,j,self)

                self.metagrid[i][j] = [[dtxi],[dtzi],[dtxj_up],[dtxj_down],[dtxj_left],[dtxj_right],[dtzj_up],[dtzj_down],[dtzj_left],[dtzj_right],[di],[dj_up],[dj_down],[dj_left],[dj_right],[n_up],[n_down],[n_left],[n_right],[pos1],[pos2],[bc],[geo]]

                '''
                self.metagrid[i][j] = [
                [n_up],[n_down],[n_left],[n_right],
                [pos1],[pos2],
                [geo]
                ]
                '''

    def _game_gen_state_condi(self):
        # Calculate strain energy
        self.model.restore()
        self.model.gen_all()
        self.strain_e = self.model.U_full
        # Calculate initial strain energy
        if self.game_step == 1:
            self.int_strain_e = self.model.U_full
        else:
            pass

    def _game_get_1_state(self,do,multi=False):
        if multi:
            pass
        else:
            self._update_metagrid()
        # do = [i,j]
        # metagrid[z,x]
        '''
        if self.game_type==0:
            print('There is no game type')
        elif  self.game_type==1:
            # Theta-ij Pos Bc(numstate = 7)
            x = self.metagrid[do[0]][do[1]][5:-1]
        elif  self.game_type==2:
            # Z/Zmax Pos Bc  (numstate = 4)
            x = self.metagrid[do[0]][do[1]][9:]
        elif  self.game_type==3:
            # Theta-ij di dj  (numstate = 9)
            x = self.metagrid[do[0]][do[1]][:9]
        elif  self.game_type==4:
            # Theta-ij di dj Z/Zmax Pos Bc  (numstate = 13)
            x = self.metagrid[do[0]][do[1]]
        '''
        x = self.metagrid[do[0]][do[1]]

        state = np.array(x)
        return state

    def _game_get_next_state(self,do,action,i=0,multi=False):
        num = [action[0][0],action[0][1],action[0][2],action[0][3],action[0][4],action[0][5]]
        adjnum = [action[0][0],action[0][1]]
        movenum = [action[0][2],action[0][3],action[0][4],action[0][5]]
        movenum = movenum.index(max(movenum))
        adjnum = adjnum.index(max(adjnum))
        num = num.index(max(num))
        # next_state = f(action)
        # Interprete action
        if adjnum == 0:
            step = action[0][0]*self.bound[0]
            if self.gen_model.n_u_name_div[do[0]][do[1]].res[1] !=1:
                if self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] + step <= self.max_y_val:
                    self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] += step
                    if self.tell_action == True:
                        print('Z+:{}'.format(action[0][0]*self.bound[0]))
                else:
                    pass
            else:
                pass
        elif adjnum == 1:
            step = action[0][1]*self.bound[1]
            if self.gen_model.n_u_name_div[do[0]][do[1]].res[1] != 1:
                if self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] - step >= 0:
                    self.gen_model.n_u_name_div[do[0]][do[1]].coord[1] -= step
                    if self.tell_action == True:
                        print('Z-:{}'.format(action[0][1]*self.bound[1]))
                else:
                    pass
            else:
                pass
        if movenum == 0:
            # move right x+1
            # do[z,x]
            if (do[1]+1 != (len(self.gen_model.n_u_name_div[0]))):
                self.doing[i][1] = do[1]+1
            else:
                pass
        elif movenum == 1:
            # move left x-1
            # do[z,x]
            if (do[1] != 0):
                self.doing[i][1] = do[1]-1
            else:
                pass
        elif movenum == 2:
            # move up z-1
            # do[z,x]
            if (do[0] != 0):
                self.doing[i][0] = do[0]-1
            else:
                pass
        elif movenum == 3:
            # move down z+1
            # do[z,x]
            if (do[0]+1 != (len(self.gen_model.n_u_name_div))):
                self.doing[i][0] = do[0]+1
            else:
                pass

        #announce1 = ['z_up','z_down']
        announce2 = ['move right','move left','move up','move down']
        if self.tell_action == True:
            #print(announce1[adjnum-1]) # print out action
            print(announce2[movenum-1]) # print out action

        if multi:
            pass
        else:
            self._update_metagrid()

            if self.game_type==0:
                print('There is no game type')
            elif  self.game_type==1:
                # Theta-ij Pos Bc(numstate = 7)
                x = self.metagrid[self.doing[i][0]][self.doing[i][1]][5:-1]
            elif  self.game_type==2:
                # Z/Zmax Pos Bc  (numstate = 4)
                x = self.metagrid[self.doing[i][0]][self.doing[i][1]][9:]
            elif  self.game_type==3:
                # Theta-ij di dj  (numstate = 9)
                x = self.metagrid[self.doing[i][0]][self.doing[i][1]][:9]
            elif  self.game_type==4:
                # Theta-ij di dj Z/Zmax Pos Bc  (numstate = 13)
                x = self.metagrid[self.doing[i][0]][self.doing[i][1]]

            next_state = np.array(x)
            return next_state

    def _game_get_next_state_maddpg(self,do,i):
        '''
        if self.game_type==0:
            print('There is no game type')
        elif  self.game_type==1:
            # Theta-ij Pos Bc(numstate = 7)
            x = self.metagrid[self.doing[i][0]][self.doing[i][1]][5:-1]
        elif  self.game_type==2:
            # Z/Zmax Pos Bc  (numstate = 4)
            x = self.metagrid[self.doing[i][0]][self.doing[i][1]][9:]
        elif  self.game_type==3:
            # Theta-ij di dj  (numstate = 9)
            x = self.metagrid[self.doing[i][0]][self.doing[i][1]][:9]
        elif  self.game_type==4:
            # Theta-ij di dj Z/Zmax Pos Bc  (numstate = 13)
            x = self.metagrid[self.doing[i][0]][self.doing[i][1]]
        '''
        x = self.metagrid[self.doing[i][0]][self.doing[i][1]]

        next_state = np.array(x)
        return next_state

    def _gen_gen_reward_condition(self):
        # Calculate next state strain energy
        self.model.restore()
        self.model.gen_all()
        self.next_strain_e = self.model.U_full

    def _game_get_reward(self,agent):

        self.reward[agent] += 1000*(self.strain_e[0]-self.next_strain_e[0])/(self.int_strain_e[0])

        if self.game_step == self.end_step:
            self.done_counter = 1

        return self.reward[agent],self.done_counter

    def reset(self):
        self.state = []
        self.action = []
        self.reward = []
        self.doing = []
        self.doing = [[1,1],[len(self.gen_model.n_u_name_div)-2,len(self.gen_model.n_u_name_div[0])-2],
                     [1,len(self.gen_model.n_u_name_div[0])-2],
                     [len(self.gen_model.n_u_name_div)-2,1]
                     ]
        for i in range(self.num_agents):
            self.reward.append(0)

        self.next_state = []
        self.done = []
        self.game_step = 1

        self.xmax = 0
        self.xmin = 0
        self.ymax = 0
        self.ymin = 0
        self.zmax = 0
        self.zmin = 0

        self.dymax = 0
        self.dymin = 0
        self.sdyval = 0
        self.dmax0 = 0
        self.dmaxset = 0
        self.dtxmax = 0
        self.dtxmin = 0
        self.dtymax = 0
        self.dtymin = 0
        self.dtzmax = 0
        self.dtzmin = 0
        self.dtxmax0 = 0
        self.dtymax0 = 0
        self.dtzmax0 = 0

        self.int_strain_e = 0
        self.strain_e = 0
        self.next_strain_e =0

        self.reward_counter = []
        for i in range(self.num_agents):
            self.reward.append(0)
            self.reward_counter.append(0)
        self.done_counter = 0


    def step(self):
        self.state = self.next_state
        self.action = []
        #self.reward = []
        for i in range(len(self.reward)):
            self.reward[i] = 0
        self.next_state = []
        self.done = []
        self.game_step += 1




'''
#=============================================================================
# GAME TEMPLATE

class game1:
    def __init__(self, num_agents =1):
        self.name = 'GAME 1'
        self.description = 'No description yet'
        self.num_agents = num_agents
        #=======================
        #State Action'
        # for MADDPG
        #=======================
        self.state = []
        self.action = [] #get action from rl or data file
        self.reward = []
        self.next_state = []
        self.done = []

    def _game_get_1_state(self):
        # state is make from some randomness
        state = None
        return state
    def _game_gen_1_state(self):
        x =[]
        for i in range(self.num_agents):
            x.append(_game_get_1_state())
        self.state.append(x)

    def _game_get_next_state(self,action=None):
        # next_state = f(action)
        next_state = None
        done = None
        return next_state,done

    def _game_gen_next_state(self):

        x = []
        y = []
        for i in range(self.num_agents):
            x.append(_game_get_next_state()[0])
            y.append(_game_get_next_state()[1])
        self.next_state.append(x)
        self.done.append(y)

    def _game_get_reward(self):
        # reward = f(state,next_state)
        reward = None
        return reward

    def _game_gen_reward(self):
        # gen reward from current state
        x = []
        for i in range(self.num_agents):
            x.append(_game_get_reward(self.state[-1][i],self.next_state[-1][i]))
        self.reward.append(x)

    def reset(self):
        self.state = []
        self.action = []
        self.reward = []
        self.next_state = []
        self.done = []

    def step(self):
        self.state = self.next_state
        self.action = []
        self.reward = []
        self.next_state = []
        self.done = []

#=============================================================================
'''










