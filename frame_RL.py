'''
=====================================================================
Reinforcement Learning Framwork File
This file contains reinforcement learning framworks, using Keras
    Q-Leaning algorithm
    Actor-Critic algorithm
    Deep Deterministic Policy Gradient (DDPG)
    Multi-agent Deep Deterministic Policy (MADDPG)
Adjustable parameter are under '研究室'

強化学習フレームワークファイル
このファイルには、Kerasを使用した強化学習フレームワークが含まれています
     Q学習アルゴリズム
     二重Q学習アルゴリズム
     Actor-Criticアルゴリズム
     Advantange Actor-Critic（A2C）アルゴリズム-（未完成）
     ディープデターミニスティックポリシーグラディエント（DDPG）
     マルチエージェントディープデターミニスティックポリシー（MADDPG）
調整可能なパラメータは「研究室」の下にあります
=====================================================================
'''
import math
import os
import datetime
import random
from collections import deque

import numpy as  np
import tensorflow as tf

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, concatenate,BatchNormalization,LeakyReLU,merge,Concatenate
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
from keras.activations import softmax
import keras.backend as K


#--------------------------------
# Q-Learning class with keras
#--------------------------------
class Q_Learning:
    def __init__(self,lr,ep,epd,gamma,q_nn,max_mem,num_ob,num_action,sess):
        self.lr = lr # Learning Rate / 学習率
        self.epint = ep # Initial epsilon value / イプシロンの初期値
        self.ep = ep # Current epsilon value / 現在のイプシロン値
        self.epd = epd # Epsilon decay value / イプシロン減衰値
        self.epmin = 0.05 # Minimum epsilon value / イプシロンの最小値 (研究室)
        self.gamma = gamma # Reward discount factor / イプシロンの最小値...
        self.q_nn = q_nn # list of neuron in each layer, len(list) = n_layers / 各層のニューロンのリスト、len（list）= n_layers
        self.temprp = deque(maxlen = max_mem) # replay buffer(memory) / 再生バッファ(メモリ)
        self.num_state = num_ob # numbers of state (neural network's input) / 状態の数（ニューラルネットワークの入力）
        self.num_action = num_action # numbers of action (neural network's output) アクションの数（ニューラルネットワークの出力）
        self.batch_size = 1 # How much memory used in a single training / 1回のトレーニングで使用されるメモリの量 (研究室)
        self.sess = sess # Tensorflow calculation session / Tensorflow計算セッション
        self.loss = [] # Loss in each training / 各トレーニングでの損失 (研究室)
        self.q_model = self.create_q_model() # Create a neural network model / ニューラルネットワークモデルを作成する
        self.sess.run(tf.compat.v1.initialize_all_variables()) # Initialize Tensorflow calculation session / Tensorflow計算セッションを初期化する

    # Function to create an agent, no input / エージェントを作成する関数、入力なし
    def create_q_model(self):
        state_input = Input(shape=tuple([self.num_state])) # Neural Network's Input layer / ニューラルネットワーク入力層
        x = Dense(self.q_nn[0],activation='relu')(state_input) # Neural Network's Hidden layer 1 / ニューラルネットワーク非表示レイヤー1
        for i in range(len(self.q_nn)-1): # Neural Network's Hidden layer i+1 / ＃ニューラルネットワーク非表示レイヤーi + 1
            x = Dense(self.q_nn[i+1],activation='relu')(x)
        output = Dense(self.num_action)(x) # Neural Network's Output layer / ニューラルネットワーク出力層
        model = Model(input=state_input, output=output) # Neural Network's Model / ニューラルネットワークモデル
        model.compile(loss="mse", optimizer=Adam(lr=self.lr)) # Neural Network's loss and optimizer for training / ニューラルネットワークの損失とトレーニングのためのオプティマイザ
        #model.summary()
        return model # This fuction return Neural Network's Model / この関数はニューラルネットワークモデルを返します

    # Function to use agent to do forward path, input = state /エージェントを使用してパスを転送する関数、入力=状態
    def act(self,state):
        state = np.array(state).reshape(1,self.num_state) # Change state into a [self.num_state x 1] vector to feed into Neural Network / 状態を[self.num_state x 1]ベクトルに変更して、ニューラルネットワークにフィードする
        # Epsilon Greedy Method : Create a random number. If current epsilon value is larger than the random number, agent act randomly
        # イプシロン貪欲メソッド：乱数を作成します。 現在のイプシロン値が乱数より大きい場合、エージェントはランダムに動作します
        if np.random.random() < self.ep: # If current epsilon value is larger than the random number / 現在のイプシロン値が乱数より大きい場合
            actlist = [] # Create a list for random action. This list will have the same dimension as Neural Network's Output ([self.num_state x 1]) / ランダムアクションのリストを作成します。 このリストは、ニューラルネットワークの出力（[self.num_state x 1]）と同じ次元になります。
            for i in range(self.num_action):
                actlist.append(random.random()) # Put random value into the list for self.num_action / ランダムな値をself.num_actionのリストに入れます
            action = np.array([actlist]).reshape((1,self.num_action)) # Change actlist into a [self.num_statex1]vector / actlistを[self.num_state x 1]ベクトルに変更します
        else: # If current epsilon value is smaller than the random number / 現在のイプシロン値が乱数より小さい場合
            action = self.q_model.predict(state) # Neural Network do the forward path / ニューラルネットワークは順方向パスを実行します
        self.ep *= self.epd # Reduce current epsilon value / 現在のイプシロン値を減らす
        if self.ep<=self.epmin: # If current epsilon value is smaller than minimum epsilon value, current epsilon value=minimum epsilon value / 現在のイプシロン値が最小イプシロン値より小さい場合、現在のイプシロン値=最小イプシロン値
            self.ep=self.epmin
        return action # This function return action / この関数はアクションを返します

    # Function to put (state, action, reward, next_state, done_counter) into agent's replay buffer(memory) / （状態、アクション、報酬、next_state、done_counter）をエージェントのリプレイバッファー（メモリ）に配置する関数。
    def remember(self, state, action, reward, next_state, done):
        self.temprp.append([state, action, reward, next_state, done])

    # Sub_Function to train agent, input = sample from memory / エージェントをトレーニングするSub_Function、入力=メモリからのサンプル
    def _train_q_model(self, samples):
        '''
        -------------------------------------------------------------------
        Trainig a Neural Network
            0. Creating training data [x,y]
            1. Forward path: Feed x into Neural Network to calculate y'
            2. Calculate Loss from y and y'
            3. Calculate dL/dw and dL/db using Backpropagation
            4. Update w and b in Neural Network
        In Q-Learning
            x is state (dimension[self.num_statex1])
            y is output(dimension[self.num_actionx1]) which has the maximum equal to (reward + gamma*maxQ(next_state))
            y' is Q-value which is Output of Neural Network(dimension[self.num_actionx1])

        ニューラルネットワークのトレーニング
             0.トレーニングデータの作成[x、y]
             1.フォワードパス：xをニューラルネットワークに入力してy 'を計算する
             2. yとy 'から損失を計算する
             3.バックプロパゲーションを使用してdL / dwおよびdL / dbを計算する
             4.ニューラルネットワークのwとbを更新する
         Qラーニング
             xは状態です（dimension [self.num_statex1]）
             yは、最大の（reward + gamma * maxQ（next_state））に等しいoutput（dimension [self.num_actionx1]）です。
             y 'はニューラルネットワークの出力であるQ値です（dimension [self.num_actionx1]）
        -------------------------------------------------------------------
        '''
        states = np.array([val[0] for val in samples]) # Extract states from memory / メモリから状態を抽出する
        next_states = np.array([(np.zeros((1,self.num_state))
                                 if val[4] is 1 else val[3].reshape(1,self.num_state)) for val in samples]) # Extract next_states from memory / メモリからnext_statesを抽出する
        q_states = self.q_model.predict_on_batch(states.reshape(-1,self.num_state)) # Use Agent to calculate Q(state) from extracted states / エージェントを使用して、抽出された状態からQ（状態）を計算する
        q_next_states = self.q_model.predict_on_batch(next_states.reshape(-1,self.num_state)) # Use Agent to calculate Q(next_state) from extracted states / エージェントを使用して、抽出された状態からQ（next_state）を計算します
        x = np.zeros((len(samples), self.num_state)) # Create list to contain x (training data) / xを含むリストを作成（トレーニングデータ）
        y = np.zeros((len(samples), self.num_action)) # Create list to contain y (training data) / yを含むリストを作成（トレーニングデータ）
        for i, b in enumerate(samples):
            state, action, reward, next_state, done = b[0], b[1], b[2], b[3], b[4]
            current_q = q_states[i] # y for this sample / このサンプルのy
            if done is 1: # if this is the end state (done counter ==1) / これが終了状態の場合（完了カウンター== 1）
                feed_act = action[0].tolist().index(max(action[0].tolist())) # find the index of maximum action / 最大アクションのインデックスを見つける
                current_q[feed_act] = reward  # change that action[index] into reward / そのアクション[インデックス]を報酬に変更します
            else: # if this is not the end state (done counter ==0) / これが終了状態でない場合（完了カウンター== 0）
                feed_act = action[0].tolist().index(max(action[0].tolist())) # find the index of maximum action / 最大アクションのインデックスを見つける
                current_q[feed_act] = reward + self.gamma * np.amax(q_next_states[i]) # change that action[index] into reward+gamma*maxQ(next_state) / そのアクション[インデックス]をreward + gamma * maxQ（next_state）に変更します
            x[i] = state.reshape(-1,self.num_state) # reshape x for training the Neural Network / ニューラルネットワークをトレーニングするためにxを再形成する
            y[i] = current_q  # y for training the Neural Network / yニューラルネットワークのトレーニング
        self.loss.append(self.q_model.train_on_batch(x, y)) # Stochastic Gradient Descent(new version of backpropagation) and Update / 確率的勾配降下法（新しいバージョンの逆伝播）と更新

    # Function to train agent, input = sample from memory / エージェントをトレーニングする関数、入力=メモリからのサンプル
    def train(self):
        batch_size = self.batch_size # How much memory used in a training / トレーニングで使用されたメモリの量
        if len(self.temprp) < batch_size: # If there is enough memory, do the training / 十分なメモリがある場合は、トレーニングを行います
            return
        samples = random.sample(self.temprp, batch_size) # Sample from memory randomly / メモリからランダムにサンプリング
        self._train_q_model(samples) # Do Sub_Function to train agent / Sub_Functionを実行してエージェントをトレーニングするs

#--------------------------------
# Actor_Critic class with keras
#--------------------------------
class Actor_Critic:
    def __init__(self,lr,ep,epd,gamma,a_nn,c_nn,max_mem,num_ob,num_action,sess):
        self.number = 1
        self.lr = lr
        self.epint = ep
        self.ep = ep
        self.epd = epd
        self.epmin=0.05
        self.gamma = gamma
        self.a_nn = a_nn
        self.c_nn = c_nn
        self.temprp = deque(maxlen = max_mem)
        self.num_state = num_ob
        self.num_action = num_action
        self.batch_size = 64
        self.tau = 0.05 # soft update
        self.sess = sess
        self.var_actor = None
        self.var_critic= None
        self.update_num =0
        self.c_loss = []

        # Actor Model
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()
        self.actor_critic_grad = tf.placeholder(tf.float32,[None, self.num_action]) # where we will feed de/dC (from critic)
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor) (-self.actor_critic_grad for gradient assent of policy function)
        # tf.gradients(ys, xs, grad_ys=None) = (grad_ys)*(dy/dx)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.lr*0.1).apply_gradients(grads)

        # Critic Model
        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.critic_target_model = self.create_critic_model()
        self.critic_grads = tf.gradients(self.critic_model.output,self.critic_action_input) # where we calcaulte de/dC for feeding above
        # Initialize for later gradient calculations
        self.sess.run(tf.compat.v1.initialize_all_variables())
        self.update_init()

    def act(self,state):
        self.ep *= self.epd
        if self.ep<=self.epmin:
            self.ep=self.epmin
        state = np.array(state).reshape(1,self.num_state)

        if np.random.random() < self.ep:
            actlist = []
            sumact=0


            for i in range(self.num_action):
                actlist.append(random.randrange(1000)/1000)
                sumact+=actlist[-1]

            for i in range(self.num_action):
                actlist[i]/=sumact
            action = np.array([actlist]).reshape((1,self.num_action))


        else:
            action = self.actor_model.predict(state)
        self.update_num += 1
        return action
        # use actor to act
    def create_actor_model(self):

        state_input = Input(shape=tuple([self.num_state]))
        x = Dense(self.a_nn[0],
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(state_input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        for i in range(len(self.a_nn)-1):
            x = Dense(self.a_nn[i+1],
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        output = Dense(self.num_action, activation='softmax')(x)

        model = Model(input=state_input, output=output)
        adam  = Adam(lr=self.lr*0.1)
        model.compile(loss="mse", optimizer=adam) #does not matter because we use grad
        self.var_actor = tf.compat.v1.global_variables_initializer()
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=tuple([self.num_state]))
        state_h1 = Dense(self.c_nn[0],
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(state_input)
        state_h1 = BatchNormalization()(state_h1)
        state_h1 = LeakyReLU()(state_h1)

        action_input = Input(shape=tuple([self.num_action]))
        action_h1    = Dense(self.c_nn[0],
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(action_input)
        action_h1 = BatchNormalization()(action_h1)
        action_h1 = LeakyReLU()(action_h1)

        x = Add()([state_h1, action_h1])
        for i in range(len(self.c_nn)-1):
            x = Dense(self.c_nn[i+1],
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        output = Dense(1)(x)

        model  = Model(input=[state_input,action_input], output=output)
        adam  = Adam(lr=self.lr)
        model.compile(loss="mse", optimizer=adam)
        self.var_critic = tf.compat.v1.global_variables_initializer()
        return state_input, action_input, model

    def remember(self, state, action, reward, next_state, done):
        self.temprp.append([state, action, reward, next_state, done])

    def _train_actor(self, samples):
        states = np.array([val[0] for val in samples])
        predicted_actions = self.actor_model.predict_on_batch(states.reshape(-1,self.num_state))
        grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input:  states.reshape(-1,self.num_state),
                self.critic_action_input: predicted_actions
            })[0]
        # calculate self.critic_grads from (critic's inputs aka state, actor's action)
        self.sess.run(self.optimize, feed_dict={
            self.actor_state_input: states.reshape(-1,self.num_state),
            self.actor_critic_grad: grads
        })
        # self.optimize does gradient ascent using grads and (self.actor_grads,actor's params)
        # train on actor
    def _train_critic(self, samples):
        states = np.array([val[0] for val in samples])

        next_states = np.array([(np.zeros((1,self.num_state))
                                 if val[4] is 1 else val[3].reshape(1,self.num_state)) for val in samples])
        target_action = self.target_actor_model.predict_on_batch(states.reshape(-1,self.num_state))
        q_s_a = self.critic_target_model.predict_on_batch([states.reshape(-1,self.num_state), target_action])
        next_target_action = self.target_actor_model.predict_on_batch(next_states.reshape(-1,self.num_state))
        q_s_a_d = self.critic_target_model.predict_on_batch([next_states.reshape(-1,self.num_state), next_target_action])
        # use target-q to calculate current_q and q_s_a_d
        x = np.zeros((len(samples), self.num_state))
        tar = np.zeros((len(samples), self.num_action))
        y = np.zeros((len(samples), 1))
        for i, b in enumerate(samples):
            state, action, reward, next_state, done = b[0], b[1], b[2], b[3], b[4]
            current_q = q_s_a[i]
            if done is 1:
                feed_act = action[0].tolist().index(max(action[0].tolist()))
                current_q[0] = reward
            else:
                feed_act = action[0].tolist().index(max(action[0].tolist()))
                current_q[0] = reward + self.gamma * np.amax(q_s_a_d[i])
            x[i] = state.reshape(-1,self.num_state)
            tar[i] = action.reshape(-1,self.num_action)
            y[i] = current_q
        self.c_loss.append(self.critic_model.train_on_batch([x, tar], y))
        # train q

    def train(self):
        batch_size = self.batch_size
        if len(self.temprp) < batch_size:
            return

        rewards = []
        samples = random.sample(self.temprp, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)
    #   Target Model Updating
    def _update_actor_target(self,init=None):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        if init==1:
            for i in range(len(actor_target_weights)):
                actor_target_weights[i] = actor_model_weights[i]
        # Softupdate using tau
        else:
            for i in range(len(actor_target_weights)):
                actor_target_weights[i] = (actor_model_weights[i]*(1-self.tau)) + (actor_target_weights[i]*self.tau)
        self.target_actor_model.set_weights(actor_target_weights) #use for train critic_model_weights

    def _update_critic_target(self,init=None):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()
        if init==1:
            for i in range(len(critic_target_weights)):
                critic_target_weights[i] = critic_model_weights[i]
        # Softupdate using tau
        else:
            for i in range(len(critic_target_weights)):
                critic_target_weights[i] = (critic_target_weights[i]*(1-self.tau)) + (critic_model_weights[i]*self.tau)
        self.critic_target_model.set_weights(critic_target_weights) #use for train critic_model_weights

    def update(self):
        # Softupdate using tau every self.update_num interval
        if self.update_num == 1000:
            self._update_actor_target()
            self._update_critic_target()
            self.update_num = 0
            print('update target')
        else:
            pass
    def update_init(self):
        self._update_actor_target(1)
        self._update_critic_target(1)


#--------------------------------
# DDPG_Actor_Critic class with keras
#--------------------------------
# Ornstein-Uhlenbeck noise
class OUNoise():
    def __init__(self, mu, theta, sigma):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = 0.001
    def gen_noise(self,x):
        return self.theta*(self.mu-x)*self.dt + self.sigma*np.random.randn(1)

class DDPG_Actor_Critic:
    def __init__(self,lr,ep,epd,gamma,a_nn,c_nn,max_mem,num_ob,num_action,sess,mu,theta,sigma):
        self.number = 1
        self.lr = lr
        self.epint = ep
        self.ep = ep
        self.epd = epd
        self.epmin=0.05
        self.gamma = gamma
        self.a_nn = a_nn
        self.c_nn = c_nn
        self.temprp = deque(maxlen = max_mem)
        self.num_state = num_ob
        self.num_action = num_action
        self.batch_size = 64
        self.tau = 0.05 # soft update
        self.sess = sess
        self.var_actor = None
        self.var_critic= None
        self.noise = []#[NoiseofAction1,NoiseofAction2,...]
        self.update_num =0
        self.c_loss = []

        self.create_noise(mu,theta,sigma)
        # Actor Model
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()
        self.actor_critic_grad = tf.placeholder(tf.float32,[None, self.num_action]) # where we will feed de/dC (from critic)
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor) (-self.actor_critic_grad for gradient assent of policy function)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.lr*0.1).apply_gradients(grads)

        # Critic Model
        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.critic_target_model = self.create_critic_model()
        self.critic_grads = tf.gradients(self.critic_model.output,self.critic_action_input) # where we calcaulte de/dC for feeding above
        # Initialize for later gradient calculations
        self.sess.run(tf.compat.v1.initialize_all_variables())
        self.update_init()


    def create_noise(self,mu, theta, sigma):
        # mu = [mu of action1,mu of action2,... ]
        # theta = [theta of action1,theta of action2,... ]
        # sigma = [sigma of action1,sigma of action2,... ]
        for i in range(self.num_action):
            self.noise.append(OUNoise(mu[i], theta[i], sigma[i]))

    def act(self,state):
        self.ep *= self.epd
        if self.ep<=self.epmin:
            self.ep=self.epmin
        state = np.array(state).reshape(1,self.num_state)
        action = self.actor_model.predict(state)
        if self.noise != None:
            for i in range(len(action[0])):
                action[0][i] += self.noise[i].gen_noise(action[0][i])
        self.update_num += 1

        return action

    def create_actor_model(self):
        state_input = Input(shape=tuple([self.num_state]))
        x = Dense(self.a_nn[0],
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(state_input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        for i in range(len(self.a_nn)-1):
            x = Dense(self.a_nn[i+1],
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        # OUTPUT NODES
        adjust1 = Dense(1,activation='sigmoid',kernel_initializer=keras.initializers.glorot_normal(seed=None))(x) # output for adjusting node's height ,output range[0,1] use sigmoid
        adjust2 = Dense(1,activation='sigmoid',kernel_initializer=keras.initializers.glorot_normal(seed=None))(x) # output for adjusting node's height ,output range[0,1] use sigmoid
        move1 = Dense(1,activation='sigmoid',kernel_initializer=keras.initializers.glorot_normal(seed=None))(x) # output for move to right node ,output range[0,1] use sigmoid
        move2 = Dense(1,activation='sigmoid',kernel_initializer=keras.initializers.glorot_normal(seed=None))(x) # output for move to left node ,output range[0,1] use sigmoid
        move3 = Dense(1,activation='sigmoid',kernel_initializer=keras.initializers.glorot_normal(seed=None))(x) # output for move to up node ,output range[0,1] use sigmoid
        move4 = Dense(1,activation='sigmoid',kernel_initializer=keras.initializers.glorot_normal(seed=None))(x) # output for move to down node ,output range[0,1] use sigmoid
        output = Concatenate()([adjust1,adjust2,move1,move2,move3,move4])

        model = Model(input=state_input, output=output)
        adam  = Adam(lr=self.lr*0.1)
        model.compile(loss="mse", optimizer=adam)
        self.var_actor = tf.compat.v1.global_variables_initializer()
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=tuple([self.num_state]))
        state_h1 = Dense(self.c_nn[0],
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(state_input)
        state_h1 = BatchNormalization()(state_h1)
        state_h1 = LeakyReLU()(state_h1)

        action_input = Input(shape=tuple([self.num_action]))
        action_h1    = Dense(self.c_nn[0],
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(action_input)
        action_h1 = BatchNormalization()(action_h1)
        action_h1 = LeakyReLU()(action_h1)

        x = Add()([state_h1, action_h1])
        for i in range(len(self.c_nn)-1):
            x = Dense(self.c_nn[i+1],
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        output = Dense(1,kernel_initializer=keras.initializers.glorot_normal(seed=None))(x)

        model  = Model(input=[state_input,action_input], output=output)
        adam  = Adam(lr=self.lr)
        model.compile(loss="mse", optimizer=adam)
        self.var_critic = tf.compat.v1.global_variables_initializer()
        return state_input, action_input, model

    def remember(self, state, action, reward, next_state, done):
        self.temprp.append([state, action, reward, next_state, done])

    def _train_actor(self, samples):
        states = np.array([val[0] for val in samples])
        predicted_actions = self.actor_model.predict_on_batch(states.reshape(-1,self.num_state))
        grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input:  states.reshape(-1,self.num_state),
                self.critic_action_input: predicted_actions
            })[0]
        #print(grads)
        self.sess.run(self.optimize, feed_dict={
            self.actor_state_input: states.reshape(-1,self.num_state),
            self.actor_critic_grad: grads
        })

    def _train_critic(self, samples):
        states = np.array([val[0] for val in samples])

        next_states = np.array([(np.zeros((1,self.num_state))
                                 if val[4] is 1 else val[3].reshape(1,self.num_state)) for val in samples])
        target_action = self.target_actor_model.predict_on_batch(states.reshape(-1,self.num_state))
        q_s_a = self.critic_target_model.predict_on_batch([states.reshape(-1,self.num_state), target_action])
        next_target_action = self.target_actor_model.predict_on_batch(next_states.reshape(-1,self.num_state))
        q_s_a_d = self.critic_target_model.predict_on_batch([next_states.reshape(-1,self.num_state), next_target_action])
        x = np.zeros((len(samples), self.num_state))
        tar = np.zeros((len(samples), self.num_action))
        y = np.zeros((len(samples), 1))
        for i, b in enumerate(samples):
            state, action, reward, next_state, done = b[0], b[1], b[2], b[3], b[4]
            current_q = q_s_a[i]
            if done is 1:
                feed_act = action[0].tolist().index(max(action[0].tolist()))
                current_q[0] = reward
            else:
                feed_act = action[0].tolist().index(max(action[0].tolist()))
                current_q[0] = reward + self.gamma * np.amax(q_s_a_d[i])
            x[i] = state.reshape(-1,self.num_state)
            tar[i] = action.reshape(-1,self.num_action)
            y[i] = current_q

        self.c_loss.append(self.critic_model.train_on_batch([x, tar], y))

    def train(self):
        batch_size = self.batch_size
        if len(self.temprp) < batch_size:
            return

        rewards = []
        samples = random.sample(self.temprp, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)
    #   Target Model Updating
    def _update_actor_target(self,init=None):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        if init==1:
            for i in range(len(actor_target_weights)):
                actor_target_weights[i] = actor_model_weights[i]
        # Softupdate using tau
        else:
            for i in range(len(actor_target_weights)):
                actor_target_weights[i] = (actor_model_weights[i]*(1-self.tau)) + (actor_target_weights[i]*self.tau)
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self,init=None):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()
        if init==1:
            for i in range(len(critic_target_weights)):
                critic_target_weights[i] = critic_model_weights[i]
        # Softupdate using tau
        else:
            for i in range(len(critic_target_weights)):
                critic_target_weights[i] = (critic_target_weights[i]*(1-self.tau)) + (critic_model_weights[i]*self.tau)
        self.critic_target_model.set_weights(critic_target_weights)

    def update(self):
        # Softupdate using tau every self.update_num interval
        if self.update_num == 1000:
            self._update_actor_target()
            self._update_critic_target()
            self.update_num = 0
            print('update target')
        else:
            pass
    def update_init(self):
        self._update_actor_target(1)
        self._update_critic_target(1)

#--------------------------------
# MADDPG_Actor_Critic class with keras
#--------------------------------
class OneAgent:
    def __init__(self,lr,ep,epd,gamma,a_nn,c_nn,num_ob,num_action,sess,mu,theta,sigma,all_agent,batch):
        self.number = 1
        self.all_agent = all_agent
        self.lr = lr
        self.epint = ep
        self.ep = ep
        self.epd = epd
        self.epmin=0.05
        self.gamma = gamma
        self.a_nn = a_nn
        self.c_nn = c_nn
        self.num_state = num_ob
        self.num_action = num_action
        self.num_critic_state_input = self.all_agent*self.num_state
        self.num_critic_action_input = self.all_agent*self.num_action
        self.batch_size = batch
        self.tau = 0.01 # soft update
        self.sess = sess
        self.var_actor = None
        self.var_critic= None
        self.noise = []#[NoiseofAction1,NoiseofAction2,...]
        self.update_num =0
        self.c_loss = []

        self.create_noise(mu,theta,sigma)
        # Actor Model
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32,[None, None]) # where we will feed de/dC (from critic)
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor) (-self.actor_critic_grad for gradient assent of policy function)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.lr*0.1).apply_gradients(grads)

        # Critic Model
        self.critic_state_input, self.critic_ot_state_input, self.critic_action_input, self.critic_ot_action_input, self.critic_model = self.create_critic_model()
        _, _,_, _, self.critic_target_model = self.create_critic_model()
        self.critic_grads = tf.gradients(self.critic_model.output,self.critic_action_input) # where we calcaulte de/dC for feeding above
        # Initialize for later gradient calculations
        self.sess.run(tf.compat.v1.initialize_all_variables())
        self.update_init()

    def create_noise(self,mu, theta, sigma):
        # mu = [mu of action1,mu of action2,... ]
        # theta = [theta of action1,theta of action2,... ]
        # sigma = [sigma of action1,sigma of action2,... ]
        for i in range(self.num_action):
            self.noise.append(OUNoise(mu[i], theta[i], sigma[i]))

    def act(self,state):
        self.ep *= self.epd
        if self.ep<=self.epmin:
            self.ep=self.epmin
        state = np.array(state).reshape(1,self.num_state)
        action = self.actor_model.predict(state)
        if self.noise != None:
            for i in range(len(action[0])):
                action[0][i] += self.noise[i].gen_noise(action[0][i])
        self.update_num += 1
        #print('ACTS')
        #print(action)

        return action

    def create_actor_model(self):
        state_input = Input(shape=tuple([self.num_state]))
        x = Dense(self.a_nn[0],
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(state_input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        for i in range(len(self.a_nn)-1):
            x = Dense(self.a_nn[i+1],
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)

        # OUTPUT NODES
        adjust1 = Dense(1,activation='sigmoid',kernel_initializer=keras.initializers.glorot_normal(seed=None))(x) # output for adjusting node's height ,output range[0,1] use sigmoid
        adjust2 = Dense(1,activation='sigmoid',kernel_initializer=keras.initializers.glorot_normal(seed=None))(x) # output for adjusting node's height ,output range[0,1] use sigmoid
        move1 = Dense(1,activation='sigmoid',kernel_initializer=keras.initializers.glorot_normal(seed=None))(x) # output for move to right node ,output range[0,1] use sigmoid
        move2 = Dense(1,activation='sigmoid',kernel_initializer=keras.initializers.glorot_normal(seed=None))(x) # output for move to left node ,output range[0,1] use sigmoid
        move3 = Dense(1,activation='sigmoid',kernel_initializer=keras.initializers.glorot_normal(seed=None))(x) # output for move to up node ,output range[0,1] use sigmoid
        move4 = Dense(1,activation='sigmoid',kernel_initializer=keras.initializers.glorot_normal(seed=None))(x) # output for move to down node ,output range[0,1] use sigmoid
        output = Concatenate()([adjust1,adjust2,move1,move2,move3,move4])

        model = Model(input=state_input, output=output)
        adam  = Adam(lr=self.lr*0.1)
        model.compile(loss="mse", optimizer=adam)
        self.var_actor = tf.compat.v1.global_variables_initializer()
        return state_input, model

    def create_critic_model(self):
        my_state_input = Input(shape=tuple([self.num_state]))
        state_h1 = Dense(round(self.c_nn[0]*0.5),
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(my_state_input)
        state_h1 = BatchNormalization()(state_h1)
        state_h1 = LeakyReLU()(state_h1)

        ot_state_input = Input(shape=tuple([self.num_critic_state_input-self.num_state]))
        state_h2 = Dense(round(self.c_nn[0]*0.5),
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(ot_state_input)
        state_h2 = BatchNormalization()(state_h2)
        state_h2 = LeakyReLU()(state_h2)

        my_action_input = Input(shape=tuple([self.num_action]))
        action_h1    = Dense(round(self.c_nn[0]*0.5),
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(my_action_input)
        action_h1 = BatchNormalization()(action_h1)
        action_h1 = LeakyReLU()(action_h1)

        ot_action_input = Input(shape=tuple([self.num_critic_action_input-self.num_action]))
        action_h2    = Dense(round(self.c_nn[0]*0.5),
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(ot_action_input)
        action_h2 = BatchNormalization()(action_h2)
        action_h2 = LeakyReLU()(action_h2)
        x1 = Add()([state_h1,action_h1])
        x2 = Add()([state_h2,action_h2])
        x = Concatenate()([x1,x2])
        #x = Concatenate()([state_h1,action_h1])
        for i in range(len(self.c_nn)-1):
            x = Dense(self.c_nn[i+1],
                kernel_initializer=keras.initializers.glorot_normal(seed=None))(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        output = Dense(1,kernel_initializer=keras.initializers.glorot_normal(seed=None))(x)

        model  = Model(input=[my_state_input,ot_state_input,my_action_input,ot_action_input], output=output)
        adam  = Adam(lr=self.lr)
        model.compile(loss="mse", optimizer=adam)
        self.var_critic = tf.compat.v1.global_variables_initializer()
        return my_state_input,ot_state_input, my_action_input,ot_action_input, model

    #   Target Model Updating
    def _update_actor_target(self,init=None):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        if init==1:
            for i in range(len(actor_target_weights)):
                actor_target_weights[i] = actor_model_weights[i]
        # Softupdate using tau
        else:
            for i in range(len(actor_target_weights)):
                actor_target_weights[i] = (actor_model_weights[i]*(1-self.tau)) + (actor_target_weights[i]*self.tau)
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self,init=None):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()
        if init==1:
            for i in range(len(critic_target_weights)):
                critic_target_weights[i] = critic_model_weights[i]
        # Softupdate using tau
        else:
            for i in range(len(critic_target_weights)):
                critic_target_weights[i] = (critic_target_weights[i]*(1-self.tau)) + (critic_model_weights[i]*self.tau)
        self.critic_target_model.set_weights(critic_target_weights)

    def update(self):
        # Softupdate using tau every self.update_num interval
        self._update_actor_target()
        self._update_critic_target()

    def update_init(self):
        self._update_actor_target(1)
        self._update_critic_target(1)

class MADDPG:
    def __init__(self,lr,ep,epd,gamma,a_nn,c_nn,max_mem,num_agents,num_ob,num_action,sess,mu,theta,sigma):
        self.num_agents = num_agents
        self.lr = lr
        self.epint = ep
        self.ep = ep
        self.epd = epd
        self.epmin=0.05
        self.gamma = gamma
        self.a_nn = a_nn
        self.c_nn = c_nn
        self.mu = mu
        self.theta =theta
        self.sigma = sigma
        self.temprp = deque(maxlen = max_mem)
        self.agents = []
        self.update_counter=[]
        self.num_state = num_ob
        self.num_action = num_action
        self.sess = sess
        self.batch_size = 256
        self.gen_agents()

    def gen_agents(self):
        agent = 'agent'
        for i in range(self.num_agents):
            self.agents.append(agent+str(i+1))
            self.agents[-1] = OneAgent(self.lr,self.ep,self.epd,self.gamma,self.a_nn,self.c_nn,self.num_state,self.num_action,self.sess,self.mu,self.theta,self.sigma,self.num_agents,self.batch_size)
            self.agents[-1].number = i+1
            self.update_counter.append(0)

    def remember(self, state, action, reward, next_state, done):
        self.temprp.append([state, action, reward, next_state, done])

    def _train_all_actor(self, samples):
        states = []
        t_act = []

        for i in range(len(self.agents)):
            states.append(np.array([val[0][i] for val in samples]))
            t_act.append(np.array([val[1][i] for val in samples]))

        states = np.array(states)
        t_act = np.array(t_act)

        for i in range(len(self.agents)):
            my_state = []
            ot_state = []
            predict_actions = []
            ot_predict_actions=[]
            for num in range(self.batch_size):
                my_state.append([])
                ot_state.append([])
                predict_actions.append([])
                ot_predict_actions.append([])
            # Original
            for j in range(len(self.agents)):
                if j==i:
                    targets = self.agents[j].target_actor_model.predict_on_batch(states[j].reshape(-1,self.num_state))
                    for num in range(self.batch_size):
                        predict_actions[num].append(targets[num])
                        my_state[num].append(states[j][num])
                else:
                    targets = t_act[j].reshape(-1,self.num_action)
                    for num in range(self.batch_size):
                        ot_predict_actions[num].append(targets[num])
                        ot_state[num].append(states[j][num])

            my_state = np.array(my_state)
            ot_state = np.array(ot_state)
            predict_actions = np.array(predict_actions)
            ot_predict_actions = np.array(ot_predict_actions)

            grads = self.sess.run(self.agents[i].critic_grads, feed_dict={
                    self.agents[i].critic_state_input:  my_state.reshape(-1,self.num_state),
                    self.agents[i].critic_ot_state_input: ot_state.reshape(-1,(self.num_agents*self.num_state)-self.num_state),
                    self.agents[i].critic_action_input: predict_actions.reshape(-1,self.num_action),
                    self.agents[i].critic_ot_action_input: ot_predict_actions.reshape(-1,(self.num_agents*self.num_action)-self.num_action)
                })[0]#[0][i*self.num_action:(i*self.num_action)+self.num_action]

            self.agents[i].sess.run(self.agents[i].optimize, feed_dict={
                self.agents[i].actor_state_input: states[i].reshape(-1,self.num_state),
                self.agents[i].actor_critic_grad: grads
            })

    def _train_all_critic(self, samples):
        states = []
        t_act = []
        next_states =[]
        for i in range(len(self.agents)):
            states.append(np.array([val[0][i] for val in samples]))
            t_act.append(np.array([val[1][i] for val in samples]))
            next_states.append(np.array([(np.zeros((1,self.num_state))
                                 if val[4] is 1 else val[3][i].reshape(1,self.num_state)) for val in samples]))
        states = np.array(states)
        t_act = np.array(t_act)
        next_states = np.array(next_states)

        for i in range(len(self.agents)):
            my_state = []
            ot_state = []
            my_next_state = []
            ot_next_state = []

            my_target_actions = []
            ot_target_actions = []
            my_next_target_action = []
            ot_next_target_action = []
            for num in range(self.batch_size):
                my_state.append([])
                ot_state.append([])
                my_next_state.append([])
                ot_next_state.append([])

                my_target_actions.append([])
                ot_target_actions.append([])
                my_next_target_action.append([])
                ot_next_target_action.append([])

            for j in range(len(self.agents)):
                # original
                if j==i:
                    targets = self.agents[j].target_actor_model.predict_on_batch(states[j].reshape(-1,self.num_state))
                    next_targets = self.agents[j].target_actor_model.predict_on_batch(next_states[j].reshape(-1,self.num_state))
                    for num in range(self.batch_size):
                        my_state[num].append(states[j][num])
                        my_next_state[num].append(next_states[j][num])
                        my_target_actions[num].append(targets[num])
                        my_next_target_action[num].append(next_targets[num])
                else:
                    targets = t_act[j].reshape(-1,self.num_action)
                    next_targets = self.agents[j].target_actor_model.predict_on_batch(next_states[j].reshape(-1,self.num_state))
                    for num in range(self.batch_size):
                        ot_state[num].append(states[j][num])
                        ot_next_state[num].append(next_states[j][num])
                        ot_target_actions[num].append(targets[num])
                        ot_next_target_action[num].append(next_targets[num])

            my_state= np.array(my_state)
            ot_state= np.array(ot_state)
            my_next_state= np.array(my_next_state)
            ot_next_state= np.array(ot_next_state)

            my_target_actions= np.array(my_target_actions)
            ot_target_actions= np.array(ot_target_actions)
            my_next_target_action= np.array(my_next_target_action)
            ot_next_target_action= np.array(ot_next_target_action)

            q_s_a = self.agents[i].critic_target_model.predict_on_batch(
                [
                my_state.reshape(-1,self.num_state),
                ot_state.reshape(-1,(self.num_agents*self.num_state)-self.num_state),
                my_target_actions.reshape(-1,self.num_action),
                ot_target_actions.reshape(-1,(self.num_agents*self.num_action)-self.num_action)
                ])

            q_s_a_d = self.agents[i].critic_target_model.predict_on_batch(
                [
                my_next_state.reshape(-1,self.num_state),
                ot_next_state.reshape(-1,(self.num_agents*self.num_state)-self.num_state),
                my_next_target_action.reshape(-1,self.num_action),
                ot_next_target_action.reshape(-1,(self.num_agents*self.num_action)-self.num_action)
                ])
            x = np.zeros((len(samples), self.num_state))
            ot_x = np.zeros((len(samples), (self.num_state*self.num_agents)-self.num_state))
            tar = np.zeros((len(samples), self.num_action))
            ot_tar = np.zeros((len(samples), (self.num_action*self.num_agents)-self.num_action))
            y = np.zeros((len(samples), 1))
            for k, b in enumerate(samples):
                state, action, reward, next_state, done = b[0], b[1], b[2], b[3], b[4]
                current_q = q_s_a[k]
                if done is 1:
                    #feed_act = action.tolist().index(max(action.tolist()))
                    current_q[0] = reward
                else:
                    #feed_act = action.tolist().index(max(action.tolist()))
                    current_q[0] = reward + self.gamma * np.amax(q_s_a_d[k])

                ot_x_all = []
                ot_tar_all = []
                for num in range(len(state)):
                    if num!=i:
                        ot_x_all.append(state[num])
                        ot_tar_all.append(action[num])

                x[k] = np.array(state[i]).reshape(-1,self.num_state)
                ot_x[k] = np.array(ot_x_all).reshape(-1,(self.num_state*self.num_agents)-self.num_state)
                tar[k] = np.array(action[i]).reshape(-1,self.num_action)
                ot_tar[k] = np.array(ot_tar_all).reshape(-1,(self.num_action*self.num_agents)-self.num_action)
                y[k] = current_q

            self.agents[i].c_loss.append(self.agents[i].critic_model.train_on_batch([x,ot_x,tar,ot_tar], y))

    def train(self):
        batch_size = self.batch_size
        if len(self.temprp) < batch_size:
            return

        rewards = []
        samples = random.sample(self.temprp, batch_size)
        self._train_all_critic(samples)
        self._train_all_actor(samples)

    def update(self):
        # Softupdate using tau every self.update_num interval
        interval = len(self.agents)*200
        if self.agents[0].update_num%interval==0:
            #if self.update_counter[0] == 0:

            for i in range(len(self.agents)):

                self.agents[i].update()
                print('update target agent{}'.format(i+1))

    def update_init(self):
        for num in range(len(self.agents)):
            self.agents[num].update_init()

