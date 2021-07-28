import tensorflow as tf
import numpy as np

K = tf.keras.backend
Optimizer = tf.keras.optimizers.Optimizer

callbacks = tf.keras.callbacks
backend = tf.keras.backend



class AdamW(Optimizer):
    """AdamW optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond".
        batch_size:       int >= 1. Train input batch size; used for normalization
        total_iterations: int >= 0. Total expected iterations / weight updates
                          throughout training, used for normalization; <1>
        weight_decays:    dict / None. Name-value pairs specifying weight decays,
                          as {<weight matrix name>:<weight decay value>}; <2>
        lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
                          multipliers, as {<layer name>:<multiplier value>}; <2>
        use_cosine_annealing: bool. If True, multiplies lr each train iteration
                              as a function of eta_min, eta_max, total_iterations,
                              and t_cur (current); [2]-Appendix, 2
        eta_min, eta_max: int, int. Min & max values of cosine annealing
                          lr multiplier; [2]-Appendix, 2
        t_cur: int. Value to initialize t_cur to - used for 'warm restarts'.
               To be used together with use_cosine_annealing==True
        init_verbose: bool. If True, print weight-name--weight-decay, and
                      lr-multiplier--layer-name value pairs set during
                      optimizer initialization (recommended)
    # <1> - if using 'warm restarts', then refers to total expected iterations
            for a given restart; can be an estimate, and training won't stop
            at iterations == total_iterations. [2]-Appendix, pg 1
    # <2> - [AdamW Keras Implementation - Github repository]
            (https://github.com/OverLordGoldDragon/keras_adamw)
    # References
        - [1][Adam - A Method for Stochastic Optimization]
             (http://arxiv.org/abs/1412.6980v8)
        - [2][Fixing Weight Decay Regularization in Adam]
             (https://arxiv.org/abs/1711.05101)
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 amsgrad=False, batch_size=32, total_iterations=0,
                 weight_decays=None, lr_multipliers=None,
                 use_cosine_annealing=False, eta_min=0, eta_max=1,
                 t_cur=0, init_verbose=True, name='AdamW', **kwargs):
        self.initial_decay = kwargs.pop('decay', 0.0)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        eta_t = kwargs.pop('eta_t', 1.)
        super(AdamW, self).__init__(name, **kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(self.initial_decay, name='decay')
            self.batch_size = K.variable(batch_size, dtype='int64',
                                         name='batch_size')
            self.total_iterations = K.variable(total_iterations, dtype='int64',
                                               name='total_iterations')
            self.eta_min = K.constant(eta_min, name='eta_min')
            self.eta_max = K.constant(eta_max, name='eta_max')
            self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
            self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')

        self.amsgrad = amsgrad
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        self.updates.append(K.update_add(self.t_cur, 1))

        lr = self.learning_rate
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p),
                      dtype=K.dtype(p),
                      name='m_' + str(i))
              for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p),
                      dtype=K.dtype(p),
                      name='v_' + str(i))
              for (i, p) in enumerate(params)]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p),
                             dtype=K.dtype(p),
                             name='vhat_' + str(i))
                     for (i, p) in enumerate(params)]
        else:
            vhats = [K.zeros(1, name='vhat_' + str(i))
                     for i in range(len(params))]
        self.weights = [self.iterations] + ms + vs + vhats

        total_iterations = K.get_value(self.total_iterations)
        if total_iterations == 0:
            print("'total_iterations'==0, must be !=0 to use "
                  + "cosine annealing and/or weight decays; "
                  + "proceeding without either")
        # Schedule multiplier
        if self.use_cosine_annealing and total_iterations != 0:
            t_frac = K.cast(self.t_cur / (self.total_iterations + 1), 'float32')
            self.eta_t = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
                         (1 + K.cos(np.pi * t_frac))
            if self.init_verbose:
                print('Using cosine annealing learning rates')
        self.lr_t = lr * self.eta_t  # for external tracking

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            # Learning rate multipliers
            multiplier_name = None
            if self.lr_multipliers:
                multiplier_name = [mult_name for mult_name in self.lr_multipliers
                                   if mult_name in p.name]
            new_lr = lr_t
            if multiplier_name:
                new_lr = new_lr * self.lr_multipliers[multiplier_name[0]]
                if self.init_verbose:
                    print('{} learning rate set for {} -- {}'.format(
                        '%.e' % K.get_value(new_lr), p.name.split('/')[0], new_lr))
            elif not multiplier_name and self.init_verbose:
                print('No change in learning rate {} -- {}'.format(
                    p.name, K.get_value(new_lr)))

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            # Weight decays
            if p.name in self.weight_decays.keys() and total_iterations != 0:
                wd = self.weight_decays[p.name]
                wd_normalized = wd * K.cast(
                    K.sqrt(self.batch_size / self.total_iterations), 'float32')
                p_t = p_t - self.eta_t * wd_normalized * p
                if self.init_verbose:
                    print('{} weight decay set for {}'.format(
                        K.get_value(wd_normalized), p.name))

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'learning_rate': float(K.get_value(self.learning_rate)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'batch_size': int(K.get_value(self.batch_size)),
                  'total_iterations': int(K.get_value(self.total_iterations)),
                  'weight_decays': self.weight_decays,
                  'lr_multipliers': self.lr_multipliers,
                  'use_cosine_annealing': self.use_cosine_annealing,
                  't_cur': int(K.get_value(self.t_cur)),
                  'eta_t': int(K.get_value(self.eta_t)),
                  'eta_min': int(K.get_value(self.eta_min)),
                  'eta_max': int(K.get_value(self.eta_max)),
                  'init_verbose': self.init_verbose,
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class LearningRateScheduler(callbacks.Callback):
    def __init__(self,
                 schedule,
                 learning_rate=None,
                 warmup=False,
                 steps_per_epoch=None,
                 verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.verbose = verbose
        self.warmup_epochs = 5 if warmup else 0
        self.warmup_steps = int(steps_per_epoch) * self.warmup_epochs if warmup else 0
        self.global_batch = 0

        if warmup and learning_rate is None:
            raise ValueError('learning_rate cannot be None if warmup is used.')
        if warmup and steps_per_epoch is None:
            raise ValueError('steps_per_epoch cannot be None if warmup is used.')

    def on_train_batch_begin(self, batch, logs=None):
        self.global_batch += 1
        if self.global_batch < self.warmup_steps:
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')
            lr = self.learning_rate * self.global_batch / self.warmup_steps
            backend.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: LearningRateScheduler warming up learning '
                      'rate to %s.' % (self.global_batch, lr))

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(backend.get_value(self.model.optimizer.lr))

        if epoch >= self.warmup_epochs:
            try:  # new API
                lr = self.schedule(epoch - self.warmup_epochs, lr)
            except TypeError:  # Support for old API for backward compatibility
                lr = self.schedule(epoch - self.warmup_epochs)
            if not isinstance(lr, (float, np.float32, np.float64)):
                raise ValueError('The output of the "schedule" function '
                                 'should be float.')
            backend.set_value(self.model.optimizer.lr, lr)

            if self.verbose > 0:
                print('\nEpoch %05d: LearningRateScheduler reducing learning '
                      'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.lr)


def poly_decay(lr=3e-4, max_epochs=100, warmup=False):
    """
    poly decay.
    :param lr: initial lr
    :param max_epochs: max epochs
    :param warmup: warm up or not
    :return: current lr
    """
    max_epochs = max_epochs - 5 if warmup else max_epochs

    def decay(epoch):
        lrate = lr * (1 - np.power(epoch / max_epochs, 0.9))
        return lrate

    return decay