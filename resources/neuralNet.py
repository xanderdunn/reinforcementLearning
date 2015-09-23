from random import uniform
import numpy as np


def initnw(layer):
    """
    Nguyen-Widrow initialization function
    :Parameters:
        layer: core.Layer object
            Initialization layer
    """
    ci = layer.ci
    cn = layer.cn
    w_fix = 0.7 * cn ** (1. / ci)
    w_rand = np.random.rand(cn, ci) * 2 - 1
    # Normalize
    if ci == 1:
        w_rand /= np.abs(w_rand)
    else:
        w_rand *= np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(cn, 1))

    w = w_fix * w_rand
    b = np.array([0]) if cn == 1 else w_fix * \
        np.linspace(-1, 1, cn) * np.sign(w[:, 0])

    # Scaleble to inp_active
    amin, amax = layer.transf.inp_active
    amin = -1 if amin == -np.Inf else amin
    amax = 1 if amax == np.Inf else amax

    x = 0.5 * (amax - amin)
    y = 0.5 * (amax + amin)
    w = x * w
    b = x * b + y

    # Scaleble to inp_minmax
    minmax = layer.inp_minmax.copy()
    minmax[np.isneginf(minmax)] = -1
    minmax[np.isinf(minmax)] = 1

    x = 2. / (minmax[:, 1] - minmax[:, 0])
    y = 1. - minmax[:, 1] * x
    w = w * x
    b = np.dot(w, y) + b

    layer.np['w'][:] = w
    layer.np['b'][:] = b

    return


def initwb_reg(layer):
    """
    Initialize weights and bias
    in the range defined by the activation function (transf.inp_active)
    :Parameters:
        layer: core.Layer object
            Initialization layer
    """
    active = layer.transf.inp_active[:]

    if np.isinf(active[0]):
        active[0] = -100.0

    if np.isinf(active[1]):
        active[1] = 100.0

    min = active[0] / (2 * layer.cn)
    max = active[1] / (2 * layer.cn)

    init_rand(layer, min, max, 'w')
    if 'b' in layer.np:
        init_rand(layer, min, max, 'b')


class Layer(object):

    """
    Abstract Neural Layer class
    :Parameters:
        ci: int
            Number of inputs
        cn: int
            Number of neurons
        co: int
            Number of outputs
        property: dict
            property: array shape
            example: {'w': (10, 1), 'b': 10}
    """

    def __init__(self, ci, cn, co, property):
        self.ci = ci
        self.cn = cn
        self.co = co
        self.np = {}
        for p, shape in property.items():
            self.np[p] = np.empty(shape)
        self.inp = np.zeros(ci)
        self.out = np.zeros(co)
        # Property must be change when init Layer
        self.out_minmax = np.empty([self.co, 2])
        # Property will be change when init Net
        self.inp_minmax = np.empty([self.ci, 2])
        self.initf = None

    def step(self, inp):
        """ Layer simulation step """
        assert len(inp) == self.ci
        out = self._step(inp)
        self.inp = inp
        self.out = out

    def init(self):
        """ Init Layer random values """
        if isinstance(self.initf, list):
            for initf in self.initf:
                initf(self)
        elif self.initf is not None:
            self.initf(self)

    def _step(self, inp):
        raise NotImplementedError("Call abstract metod Layer._step")


class Perceptron(Layer):

    """
    Perceptron Layer class
    :Parameters:
        ci: int
            Number of input
        cn: int
            Number of neurons
        transf: callable
            Transfer function
    :Example:
        >>> import neurolab as nl
        >>> # create layer with 2 inputs and 4 outputs(neurons)
        >>> l = Perceptron(2, 4, nl.trans.PureLin())
    """

    def __init__(self, ci, cn, transf):

        Layer.__init__(self, ci, cn, cn, {'w': (cn, ci), 'b': cn})

        self.transf = transf
        if not hasattr(transf, 'out_minmax'):
            test = np.asfarry([-1e100, -100, -10, -1, 0, 1, 10, 100, 1e100])
            val = self.transf(test)
            self.out_minmax = np.array([val.min(), val.max()] * self.co)
        else:
            self.out_minmax = np.asfarray([transf.out_minmax] * self.co)
        # default init function
        self.initf = initwb_reg
        self.s = np.zeros(self.cn)

    def _step(self, inp):
        self.s = np.sum(self.np['w'] * inp, axis=1)
        self.s += self.np['b']
        return self.transf(self.s)


class TanSig:

    """
    Hyperbolic tangent sigmoid transfer function
    :Parameters:
        x: ndarray
            Input array
    :Returns:
        y : ndarray
            The corresponding hyperbolic tangent values.
    :Example:
        >>> f = TanSig()
        >>> f([-np.Inf, 0.0, np.Inf])
        array([-1.,  0.,  1.])
    """
    # output range
    out_minmax = [-1, 1]
    # input active range
    inp_active = [-2, 2]

    def __call__(self, x):
        return np.tanh(x)

    def deriv(self, x, y):
        """
        Derivative of transfer function TanSig
        """
        return 1.0 - np.square(y)


class PureLin:

    """
    Linear transfer function
    :Parameters:
        x: ndarray
            Input array
    :Returns:
        y : ndarray
            copy of x
    :Example:
        >>> import numpy as np
        >>> f = PureLin()
        >>> x = np.array([-100., 50., 10., 40.])
        >>> f(x).tolist()
        [-100.0, 50.0, 10.0, 40.0]
    """

    out_minmax = [-np.Inf, np.Inf]
    inp_active = [-np.Inf, np.Inf]

    def __call__(self, x):
        return x.copy()

    def deriv(self, x, y):
        """
        Derivative of transfer function PureLin
        """
        return np.ones_like(x)


def newff(minmax, size, transf=None):
    """
    Create multilayer perceptron
    :Parameters:
        minmax: list of list, the outer list is the number of input neurons,
                        inner lists must contain 2 elements: min and max
            Range of input value
        size: the length of list equal to the number of layers
             except input layer, the element of the list is the neuron number
             for corresponding layer
            Contains the number of neurons for each layer
        transf: list (default TanSig)
            List of activation function for each layer
    :Returns:
        net: Net
    :Example:
        >>> # create neural net with 2 inputs
                >>> # input range for each input is [-0.5, 0.5]
                >>> # 3 neurons for hidden layer, 1 neuron for output
                >>> # 2 layers including hidden layer and output layer
        >>> net = newff([[-0.5, 0.5], [-0.5, 0.5]], [3, 1])
        >>> net.ci
        2
        >>> net.co
        1
        >>> len(net.layers)
        2
    """

    net_ci = len(minmax)
    net_co = size[-1]

    if transf is None:
        transf = [trans.TanSig()] * len(size)
    assert len(transf) == len(size)

    layers = []
    for i, nn in enumerate(size):
        layer_ci = size[i - 1] if i > 0 else net_ci
        l = Perceptron(layer_ci, nn, transf[i])
        l.initf = initnw
        layers.append(l)
    connect = [[i - 1] for i in range(len(layers) + 1)]

    net = Net(minmax, net_co, layers, connect, trainer(TrainBFGS), SSE())
    return net


class SSE:

    """
    Sum squared error function
    :Parameters:
        target: ndarray
            target values for network
        output: ndarray
            simulated output of network
    :Returns:
        v: float
            Error value
    """

    def __call__(self, target, output):
        e = target - output
        v = 0.5 * np.sum(np.square(e))
        return v

    def deriv(self, target, output):
        """
        Derivative of SSE error function
        :Parameters:
            target: ndarray
                target values for network
            output: ndarray
                simulated output of network
        :Returns:
            d: ndarray
                Derivative: dE/d_out
        """

        return target - output


def trainer(Train, **kwargs):
    """ Trainner init """
    c = Trainer(Train, **kwargs)
    c.__doc__ = Train.__doc__
    c.__name__ = Train.__name__
    c.__module__ = Train.__module__
    return c


class Trainer(object):

    """
    Control of network training
    """

    def __init__(self, Train, epochs=500, goal=0.01, show=100, **kwargs):
        """
        :Parameters:
            Train: Train instance
                Train algorithm
            epochs: int (default 500)
                Number of train epochs
            goal: float (default 0.01)
                The goal of train
            show: int (default 100)
                Print period
            **kwargs: dict
                other Train parametrs
        """

        # Sets defaults train params
        self._train_class = Train
        self.defaults = {}
        self.defaults['goal'] = goal
        self.defaults['show'] = show
        self.defaults['epochs'] = epochs
        self.defaults['train'] = kwargs
        if Train.__init__.__defaults__:
            # cnt = Train.__init__.func_code.co_argcount
            cnt = Train.__init__.__code__.co_argcount
            # names = Train.__init__.func_code.co_varnames
            names = Train.__init__.__code__.co_varnames
            vals = Train.__init__.__defaults__
            st = cnt - len(vals)
            for k, v in zip(names[st: cnt], vals):
                if k not in self.defaults['train']:
                    self.defaults['train'][k] = v

        self.params = self.defaults.copy()
        self.error = []

    def __str__(self):
        return 'Trainer(' + self._train_class.__name__ + ')'

    def __call__(self, net, input, target=None, **kwargs):
        """
        Run train process
        :Parameters:
            net: Net instance
                network
            input: array like (l x net.ci)
                train input patterns
            target: array like (l x net.co)
                train target patterns - only for train with teacher
            **kwargs: dict
                other Train parametrs
        """

        self.params = self.defaults.copy()
        self.params['train'] = self.defaults['train'].copy()
        for key in kwargs:
            if key in self.params:
                self.params[key] = kwargs[key]
            else:
                self.params['train'][key] = kwargs[key]

        args = []
        input = np.asfarray(input)
        assert input.ndim == 2
        assert input.shape[1] == net.ci
        args.append(input)
        if target is not None:
            target = np.asfarray(target)
            assert target.ndim == 2
            assert target.shape[1] == net.co
            assert target.shape[0] == input.shape[0]
            args.append(target)

        def epochf(err, net, *args):
            """Need call on each epoch"""
            if err is None:
                err = train.error(net, *args)
            self.error.append(err)
            epoch = len(self.error)
            show = self.params['show']
            if show and (epoch % show) == 0:
                print("Epoch: {0}; Error: {1};".format(epoch, err))
            if err < self.params['goal']:
                raise TrainStop('The goal of learning is reached')
            if epoch >= self.params['epochs']:
                raise TrainStop(
                    'The maximum number of train epochs is reached')

        train = self._train_class(net, *args, **self.params['train'])
        Train.__init__(train, epochf, self.params['epochs'])
        self.error = []
        try:
            train(net, *args)
        except TrainStop as msg:
            if self.params['show']:
                print(msg)
        else:
            if self.params['show'] and len(
                    self.error) >= self.params['epochs']:
                print("The maximum number of train epochs is reached")
        return self.error


class NeuroLabError(Exception):
    pass


class TrainStop(NeuroLabError):
    pass


class Train(object):

    """Base train abstract class"""

    def __init__(self, epochf, epochs):
        self.epochf = epochf
        self.epochs = epochs

    def __call__(self, net, *args):
        for epoch in range(self.epochs):
            err = self.error(net, *args)
            self.epochf(err, net, *args)
            self.learn(net, *args)

    def error(self, net, input, target, output=None):
        """Only for train with teacher"""
        if output is None:
            output = net.sim(input)
        return net.errorf(target, output)


def np_size(net):
    """
    Calculate count of al network parameters (weight, bias, etc...)
    """

    size = 0
    for l in net.layers:
        for prop in l.np.values():
            size += prop.size
    return size


def np_get_ref(net):
    """
    Get all network parameters in one array as reference
    Change array -> change networks
    :Example:
    >>> import neurolab as nl
    >>> net = nl.net.newff([[-1, 1]], [3, 1])
    >>> x = np_get_ref(net)
    >>> x.fill(10)
    >>> net.layers[0].np['w'].tolist()
    [[10.0], [10.0], [10.0]]
    """
    size = np_size(net)
    x = np.empty(size)
    st = 0
    for l in net.layers:
        for k, v in l.np.items():
            x[st: st + v.size] = v.flatten()
            l.np[k] = x[st: st + v.size]
            l.np[k].shape = v.shape
            st += v.size
    return x


def ff_grad_step(net, out, tar, grad=None):
    """
    Calc gradient with backpropogete method,
    for feed-forward neuron networks on each step
    :Parametrs:
        net: Net
            Feed-forward network
        inp: array, size = net.ci
            Input array
        tar: array, size = net.co
            Train target
        deriv: callable
            Derivative of error function
        grad: list of dict default(None)
            Grad on previous step
    :Returns:
        grad: list of dict
            Gradient of net for each layer,
            format:[{'w':..., 'b':...},{'w':..., 'b':...},...]
    """
    delt = [None] * len(net.layers)
    if grad is None:
        grad = []
        for i, l in enumerate(net.layers):
            grad.append({})
            for k, v in l.np.items():
                grad[i][k] = np.zeros(v.shape)

    # for output layer
    ln = len(net.layers) - 1
    layer = net.layers[ln]
    delt[ln] = net.errorf.deriv(tar, out) * layer.transf.deriv(layer.s, out)
    delt[ln] = np.negative(delt[ln])
    delt[ln].shape = delt[ln].size, 1
    grad[ln]['w'] += delt[ln] * layer.inp
    grad[ln]['b'] += delt[ln].reshape(delt[ln].size)

    bp = range(len(net.layers) - 2, -1, -1)
    for ln in bp:
        layer = net.layers[ln]
        next = ln + 1

        dS = np.sum(net.layers[next].np['w'] * delt[next], axis=0)
        delt[ln] = dS * layer.transf.deriv(layer.s, layer.out)
        delt[ln].shape = delt[ln].size, 1

        grad[ln]['w'] += delt[ln] * layer.inp
        grad[ln]['b'] += delt[ln].reshape(delt[ln].size)
    return grad


def ff_grad(net, input, target):
    """
    Calc and accumulate gradient with backpropogete method,
    for feed-forward neuron networks on each step
    :Parameters:
        net: Net
            Feed-forward network
        input: array, shape = N,net.ci
            Input array
        target: array, shape = N,net.co
            Train target
        deriv: callable
            Derivative of error function
    :Returns:
        grad: list of dict
            Gradient of net for each layer,
            format:[{'w':..., 'b':...},{'w':..., 'b':...},...]
        grad_flat: array
            All neurons property's in 1 array (reference of grad)
            It link to grad (changes grad is changes grad_flat)
        output: array
            output of network
    """
    # Init grad and link to grad_falt
    grad = []
    grad_flat = np.zeros(np_size(net))
    st = 0
    for i, l in enumerate(net.layers):
        grad.append({})
        for k, v in l.np.items():
            grad[i][k] = grad_flat[st: st + v.size]
            grad[i][k].shape = v.shape
            st += v.size
    output = []
    # Calculate grad for all batch
    for inp, tar in zip(input, target):
        out = net.step(inp)
        ff_grad_step(net, out, tar, grad)
        output.append(out)
    return grad, grad_flat, np.row_stack(output)


class TrainSO(Train):

    """
    Train class Based on scipy.optimize
    """

    def __init__(self, net, input, target, rr=0, **kwargs):
        self.net = net
        self.input = input
        self.target = target
        self.kwargs = kwargs
        self.x = np_get_ref(net)
        self.lerr = 1e10
        self.rr = rr

    def grad(self, x):
        self.x[:] = x
        g, g_flat, output = ff_grad(self.net, self.input, self.target)
        if self.rr:
            # g_flat is link to g
            tool.reg_grad(g, self.net, self.rr)
        return g_flat

    def fcn(self, x):
        self.x[:] = x
        err = self.error(self.net, self.input, self.target)
        if self.rr:
            err = tool.reg_error(err, self.net, self.rr)
        self.lerr = err
        return err

    def step(self, x):
        self.epochf(self.lerr, self.net, self.input, self.target)

    def __call__(self, net, input, target):
        raise NotImplementedError("Call abstract metod __call__")


class TrainBFGS(TrainSO):

    """
    BroydenFletcherGoldfarbShanno (BFGS) method
    Using scipy.optimize.fmin_bfgs
    :Support networks:
        newff (multi-layers perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train
        rr float (defaults 0.0)
            Regularization ratio
            Must be between {0, 1}
    """

    def __call__(self, net, input, target):
        from scipy.optimize import fmin_bfgs
        if 'disp' not in self.kwargs:
            self.kwargs['disp'] = 0
        self.kwargs['maxiter'] = self.epochs

        x = fmin_bfgs(self.fcn,
                      self.x.copy(),
                      fprime=self.grad,
                      callback=self.step,
                      **self.kwargs)
        self.x[:] = x


class Net(object):

    """
    Neural Network class
    :Parameters:
        inp_minmax: minmax: list ci x 2
            Range of input value
        co: int
            Number of output
        layers: list of Layer
            Network layers
        connect: list of list
            Connection scheme of layers*
        trainf: callable
            Train function
        errorf: callable
            Error function with derivative
    :Connect format:
        Example 1: for two-layers feed forwad network
            >>> connect = [[-1], # - layer 0 receives the input network signal;
            ...            [0],  # - layer 1 receives the output signal
            ...                  # from the layer 0;
            ...            [1]]  # - the network exit receives the output
            ...                  # signal from the layer 1.
        Example 2: for two-layers Elman network with derivatives:
            >>> connect = [[-1, 0], # - layer 0 receives the input network
            ...                     # signal and output signal from layer 0;
            ...            [0],     # - layer 1 receives the output
            ...                     # signal from the layer 0;
            ...            [1]]     # - the network exit receives the output
            ...                     # signals from the layer 1.
        """

    def __init__(self, inp_minmax, co, layers, connect, trainf, errorf):
        self.inp_minmax = np.asfarray(inp_minmax)
        self.out_minmax = np.zeros([co, 2])
        self.ci = self.inp_minmax.shape[0]
        self.co = co
        self.layers = layers
        self.trainf = trainf
        self.errorf = errorf
        self.inp = np.zeros(self.ci)
        self.out = np.zeros(self.co)
        # Check connect format
        assert self.inp_minmax.ndim == 2
        assert self.inp_minmax.shape[1] == 2
        if len(connect) != len(layers) + 1:
            raise ValueError("Connect error")
        # Check connect links
        tmp = [0] * len(connect)
        for con in connect:
            for s in con:
                if s != -1:
                    tmp[s] += 1
        for l, c in enumerate(tmp):
            if c == 0 and l != len(layers):
                raise ValueError("Connect error: Lost the signal " +
                                 "from the layer " + str(l - 1))
        self.connect = connect

        # Set inp_minmax for all layers
        for nl, nums_signal in enumerate(self.connect):
            if nl == len(self.layers):
                minmax = self.out_minmax
            else:
                minmax = self.layers[nl].inp_minmax
            ni = 0
            for ns in nums_signal:
                t = self.layers[ns].out_minmax if ns != -1 else self.inp_minmax
                if ni + len(t) > len(minmax):
                    raise ValueError("Connect error: on layer " + str(l - 1))
                minmax[ni: ni + len(t)] = t
                ni += len(t)
            if ni != len(minmax):
                raise ValueError("Connect error: Empty inputs on layer " +
                                 str(l - 1))
        self.init()

    def step(self, inp):
        """
        Simulated step
        :Parameters:
            inp: array like
                Input vector
        :Returns:
            out: array
                Output vector
        """
        # TODO: self.inp=np.asfarray(inp)?

        self.inp = inp
        for nl, nums in enumerate(self.connect):
            if len(nums) > 1:
                signal = []
                for ns in nums:
                    s = self.layers[ns].out if ns != -1 else inp
                    signal.append(s)
                signal = np.concatenate(signal)
            else:
                ns = nums[0]
                signal = self.layers[ns].out if ns != -1 else inp
            if nl != len(self.layers):
                self.layers[nl].step(signal)
        self.out = signal
        return self.out

    def sim(self, input):
        """
        Simulate a neural network
        :Parameters:
            input: array like
                array input vectors
        :Returns:
            outputs: array like
                array output vectors
        """
        input = np.asfarray(input)
        assert input.ndim == 2
        assert input.shape[1] == self.ci

        output = np.zeros([len(input), self.co])

        for inp_num, inp in enumerate(input):
            output[inp_num, :] = self.step(inp)

        return output

    def init(self):
        """
        Iinitialization layers
        """
        for layer in self.layers:
            layer.init()

    def train(self, *args, **kwargs):
        """
        Train network
        see net.trainf.__doc__
        """
        return self.trainf(self, *args, **kwargs)

    def reset(self):
        """
        Clear of deley
        """
        self.inp.fill(0)
        self.out.fill(0)
        for layer in self.layers:
            layer.inp.fill(0)
            layer.out.fill(0)

    def save(self, fname):
        """
        Save network on file
        :Parameters:
            fname: file name
        """
        tool.save(self, fname)

    def copy(self):
        """
        Copy network
        """
        import copy
        cnet = copy.deepcopy(self)

        return cnet


net = newff([[-5.0, 5.0]], [200, 1], [TanSig(), PureLin()])

for i in range(200000):
    input = uniform(-5.0, 5.0)
    x = np.full((1, 1), input)
    y = np.sin(x) * 5.0
    net.train(x, y, epochs=1, show=0, goal=0.02)

maxDifference = 0.0
numOutOfRange = 0
for i in range(10000):
    input = uniform(-4.0, 4.0)
    x = np.full((1, 1), input)
    y = np.sin(x) * 5.0
    prediction = net.sim(x)
    withinRange = False
    difference = abs(prediction - y)
    if difference > maxDifference:
        maxDifference = difference
    if prediction < y + 0.05 and prediction > y - 0.05:
        withinRange = True
    else:
        numOutOfRange += 1
        print("With input {} got prediction {} with actual {}".format(x, prediction, y))
print("maxDifference = {}".format(maxDifference))
print("{}/10000 were out of range".format(numOutOfRange))
