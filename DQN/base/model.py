import tensorflow as tf
import math


class DQN(tf.keras.Model):
    def __init__(self, n_action, n_states, cfg):
        super(DQN, self).__init__()
        self.n_actions = n_action
        self.n_states = n_states
        self.cfg = cfg
        self.layers = [tf.keras.layers.Dense(self.cfg.hidden_dims, activation='relu') for _ in
                       range(self.cfg.num_layers)]
        self.output_layer = tf.keras.layers.Dense(self.n_actions, activation='linear')

    def call(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        out = self.output_layer(x)
        return out


class DuelingDQN(tf.keras.Model):
    def __init__(self, n_action, n_states, cfg):
        super(DuelingDQN, self).__init__()
        self.n_actions = n_action
        self.n_states = n_states
        self.cfg = cfg

        self.layers = [tf.keras.layers.Dense(self.cfg.hidden_dims, activation='relu') for _ in
                       range(self.cfg.num_layers)]
        self.a_layer = tf.keras.layers.Dense(self.n_actions, activation='linear')
        self.v_layer = tf.keras.layers.Dense(1, activation='linear')

    def call(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        a = self.a_layer(x)
        v = self.v_layer(x)
        out = v + a - tf.reduce_mean(a)
        return out


class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, std_init=0.4):
        super(NoisyDense, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.std_init = std_init

        self.weight_mu = self.add_weight(shape=(input_dim, output_dim),
                                         initializer=tf.initializers.RandomUniform(minval=-1 / math.sqrt(input_dim),
                                                                                   maxval=1 / math.sqrt(input_dim)),
                                         trainable=True)
        self.weight_sigma = self.add_weight(shape=(input_dim, output_dim),
                                            initializer=tf.keras.initializers.Constant(
                                                value=std_init / math.sqrt(input_dim)),
                                            trainable=True)
        self.weight_epsilon = tf.Variable(initial_value=tf.random.normal((input_dim, output_dim)),
                                          trainable=False)

        self.bias_mu = self.add_weight(shape=(output_dim,),
                                       initializer=tf.initializers.RandomUniform(minval=-1 / math.sqrt(output_dim),
                                                                                 maxval=1 / math.sqrt(output_dim)),
                                       trainable=True)
        self.bias_sigma = self.add_weight(shape=(output_dim,),
                                          initializer=tf.keras.initializers.Constant(
                                              value=std_init / math.sqrt(output_dim)),
                                          trainable=True)
        self.bias_epsilon = tf.Variable(initial_value=tf.random.normal((output_dim,)),
                                        trainable=False)

        self.reset_noise()

    def call(self, x, training=None, **kwargs):
        print(training)
        if training:
            weight = self.weight_mu + self.weight_sigma * tf.cast(self.weight_epsilon, dtype=tf.float32)
            bias = self.bias_mu + self.bias_sigma * tf.cast(self.bias_epsilon, dtype=tf.float32)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        output = tf.matmul(x, weight) + bias
        return output

    def reset_noise(self):
        self.weight_epsilon.assign(tf.random.normal(self.weight_epsilon.shape))
        self.bias_epsilon.assign(tf.random.normal(self.bias_epsilon.shape))


class NoisyDQN(tf.keras.Model):
    def __init__(self, num_actions, num_states, cfg, sigma_init=0.5):
        super(NoisyDQN, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.cfg = cfg

        self.layers = [tf.keras.layers.Dense(self.cfg.hidden_dims, activation='relu') for _ in
                       range(self.cfg.num_layers)]
        self.fc1 = NoisyDense(self.cfg.hidden_dims, self.cfg.hidden_dims, sigma_init)
        self.fc2 = NoisyDense(self.cfg.hidden_dims, self.num_actions, sigma_init)

    def call(self, x, training=None, **kwargs):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        out = tf.nn.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()


class NoisyDuelingDQN(tf.keras.Model):
    def __init__(self, num_actions, num_states, cfg, sigma_init=0.5):
        super(NoisyDuelingDQN, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.cfg = cfg

        self.layers = [tf.keras.layers.Dense(self.cfg.hidden_dims, activation='relu') for _ in
                       range(self.cfg.num_layers)]
        self.a_layer1 = NoisyDense(self.cfg.hidden_dims, self.cfg.hidden_dims, sigma_init)
        self.a_layer2 = NoisyDense(self.cfg.hidden_dims, self.num_actions, sigma_init)

        self.v_layer1 = NoisyDense(self.cfg.hidden_dims, self.cfg.hidden_dims, sigma_init)
        self.v_layer2 = NoisyDense(self.cfg.hidden_dims, 1, sigma_init)

    def call(self, x, training=None, **kwargs):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        a = tf.nn.relu(self.a_layer1(x))
        a = self.a_layer2(a)
        v = tf.nn.relu(self.v_layer1(x))
        v = self.v_layer2(v)
        out = v + a - tf.reduce_mean(a)
        return out

    def reset_noise(self):
        self.a_layer1.reset_noise()
        self.a_layer2.reset_noise()
        self.v_layer1.reset_noise()
        self.v_layer2.reset_noise()


class MyDQN(tf.keras.Model):
    def __init__(self, n_action, n_states, cfg):
        super(MyDQN, self).__init__()
        self.n_actions = n_action
        self.n_states = n_states
        self.cfg = cfg
        self.hidden_layers = [tf.keras.layers.Dense(self.cfg.hidden_dims, activation='tanh') for _ in
                              range(self.cfg.num_layers)]
        if self.cfg.is_dueling and self.cfg.is_noisy:
            self.a_layer1 = NoisyDense(self.cfg.hidden_dims, self.cfg.hidden_dims, self.cfg.sigma_init)
            self.a_layer2 = NoisyDense(self.cfg.hidden_dims, self.n_actions, self.cfg.sigma_init)

            self.v_layer1 = NoisyDense(self.cfg.hidden_dims, self.cfg.hidden_dims, self.cfg.sigma_init)
            self.v_layer2 = NoisyDense(self.cfg.hidden_dims, 1, self.cfg.sigma_init)
        elif self.cfg.is_dueling:
            self.a_layer = tf.keras.layers.Dense(self.n_actions, activation='linear')
            self.v_layer = tf.keras.layers.Dense(1, activation='linear')
        elif self.cfg.is_noisy:
            self.output_layer1 = NoisyDense(self.cfg.hidden_dims, self.cfg.hidden_dims, self.cfg.sigma_init)
            self.output_layer2 = NoisyDense(self.cfg.hidden_dims, self.n_actions, self.cfg.sigma_init)
        else:
            self.output_layer = tf.keras.layers.Dense(self.n_actions, activation='linear')

    def call(self, x, training=True, **kwargs):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
        if self.cfg.is_dueling and self.cfg.is_noisy:
            a = tf.nn.relu(self.a_layer1(x))
            a = self.a_layer2(a)
            v = tf.nn.relu(self.v_layer1(x))
            v = self.v_layer2(v)
            out = v + a - tf.reduce_mean(a)
        elif self.cfg.is_dueling:
            a = self.a_layer(x)
            v = self.v_layer(x)
            out = v + a - tf.reduce_mean(a)
        elif self.cfg.is_noisy:
            out = self.output_layer1(x)
            out = tf.nn.relu(out)
            out = self.output_layer2(out)
        else:
            out = self.output_layer(x)
        return out

    def reset_noise(self):
        if self.cfg.is_dueling:
            self.a_layer1.reset_noise()
            self.a_layer2.reset_noise()
            self.v_layer1.reset_noise()
            self.v_layer2.reset_noise()
        else:
            self.output_layer1.reset_noise()
            self.output_layer2.reset_noise()
