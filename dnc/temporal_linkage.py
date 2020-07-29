import tensorflow as tf
import collections
import graph_utils
tf.disable_v2_behavior()

TemporalLinkageState = collections.namedtuple('TemporalLinkageState', ('link', 'precedence_weights'))

class TemporalLinkage:
    """Keeps track of write order for forward and backward addressing.
    This is a pseudo-RNNCore module, whose state is a pair 
    `(link, precedence_weights)`, where `link` is a (collection of) graphs for 
    write heads 
    (represented by a tensor with values in the range [0, 1]), 
    and `precedence_weights` records the "previous write locations" used to build the link graphs.
    The function `directional_read_weights` computes addresses following the
    forward and backward directions in the link graphs.
    """
    def __init__(self, memory_size, num_writes, save_state, batch_size, name='temporal_linkage'):
        self._memory_size = memory_size
        self._num_writes = num_writes
        self.save_state = save_state

        if self.save_state:
            with tf.variable_scope(name):
                self._link_state = graph_utils.get_variable(
                    '_link_state', 
                    shape=[batch_size, self._num_writes, self._memory_size, self._memory_size], 
                    dtype=tf.float32, 
                    initializer=tf.zeros_initializer, 
                    trainable=False, 
                    save_var=False, 
                    add_histograms=False
                )
                self._precedence_weights_state = graph_utils.get_variable(
                    '_precedence_weights_state', 
                    shape=[batch_size, 
                    self._num_writes, 
                    self._memory_size], 
                    dtype=tf.float32, 
                    initializer=tf.zeros_initializer, 
                    trainable=False, 
                    save_var=False, 
                    add_histograms=False
                )
    @property
    def state_size(self):
        return TemporalLinkageState(
            #link=tf.TensorShape([self._num_writes, self._memory_size, self._memory_size]), 
            #precedence_weights=tf.TensorShape([self._num_writes, self._memory_size]),
            link=self._num_writes * self._memory_size * self._memory_size, 
            precedence_weights=self._num_writes * self._memory_size,
            
        )

    def update_state(self, final_state):
        assert isinstance(final_state, TemporalLinkageState)
        dependencies = [
            tf.assign(self._link_state, final_state.link),
            tf.assign(self._precedence_weights_state, final_state.precedence_weights)
        ] if self.save_state else []
        with tf.control_dependencies(dependencies):
            return tf.no_op()
    
    def reset_state(self):
        dependencies = [
            tf.assign(self._link_state, tf.zeros_like(self._link_state)),
            tf.assign(self._precedence_weights_state, tf.zeros_like(self._precedence_weights_state))
        ] if self.save_state else []
        with tf.control_dependencies(dependencies):
            return tf.no_op()

    def zero_state(self, batch_size, dtype):
        if self.save_state:
            return TemporalLinkageState(link=self._link_state, precedence_weights=self._precedence_weights_state)
        else:    
            return TemporalLinkageState(
                link=tf.zeros([batch_size, self._num_writes, self._memory_size, self._memory_size]), 
                precedence_weights=tf.zeros([batch_size, self._num_writes, self._memory_size]),
            )
        

    def __call__(self, write_weights, prev_state):
        """Calculate the updated linkage state given the write weights.
        Args:
        write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
            containing the memory addresses of the different write heads.
        prev_state: `TemporalLinkageState` tuple containg a tensor `link` of
            shape `[batch_size, num_writes, memory_size, memory_size]`, and a
            tensor `precedence_weights` of shape `[batch_size, num_writes,
            memory_size]` containing the aggregated history of recent writes.
        Returns:
            A `TemporalLinkageState` tuple `next_state`, which contains the updated
            link and precedence weights.
        """
        link = self._link(prev_state.link, prev_state.precedence_weights, write_weights)
        precedence_weights = self._precedence_weights(prev_state.precedence_weights, write_weights)
        return TemporalLinkageState(link=link, precedence_weights=precedence_weights)

    def directional_read_weights(self, link, prev_read_weights, forward):
        """Calculates the forward or the backward read weights.
        For each read head (at a given address), there are `num_writes` link graphs to follow. 
        Thus this function computes a read address for each of the `num_reads * num_writes` pairs of read and write heads.
        Args:
            link: 
                `[batch_size, num_writes, memory_size, memory_size]` the link graphs L_t.
            prev_read_weights: 
                `[batch_size, num_reads, memory_size]` w_{t-1}^r.
            forward: 
                whether to follow the "future" direction in the link graph (True) or the "past" direction (False).
        Returns:
            tensor `[batch_size, num_reads, num_writes, memory_size]`
        """
        with tf.name_scope('directional_read_weights'):
            # We calculate the forward and backward directions for each pair of
            # read and write heads; hence we need to tile the read weights and do a
            # sort of "outer product" to get this.
            expanded_read_weights = tf.stack([prev_read_weights] * self._num_writes, 1)
            result = tf.matmul(expanded_read_weights, link, adjoint_b=forward)
            # Swap dimensions 1, 2 so order is [batch, reads, writes, memory]:
            return tf.transpose(result, perm=[0, 2, 1, 3])

    def _link(self, prev_link, prev_precedence_weights, write_weights):
        """Calculates the new link graphs.
        For each write head, the link is a directed graph 
        (represented by a matrix with entries in range [0, 1]) 
        whose vertices are the memory locations, and an edge indicates temporal ordering of writes.
        Args:
            prev_link: `[batch_size, num_writes, memory_size, memory_size]` 
                previous link graphs for each write head.
            prev_precedence_weights: `[batch_size, num_writes, memory_size]` 
                which is the previous "aggregated" write weights for each write head.
            write_weights: `[batch_size, num_writes, memory_size]`
                containing the new locations in memory written to.
        Returns:
            new link graphs for each write head `[batch_size, num_writes, memory_size, memory_size]`
        """
        with tf.name_scope('link'):
            batch_size = tf.shape(prev_link)[0]
            write_weights_i = tf.expand_dims(write_weights, 3)
            write_weights_j = tf.expand_dims(write_weights, 2)
            prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights, 2)
            prev_link_scale = 1 - write_weights_i - write_weights_j
            new_link = write_weights_i * prev_precedence_weights_j
            link = prev_link_scale * prev_link + new_link
            # Return the link with the diagonal set to zero, to remove self-looping edges.
            return tf.matrix_set_diag(link, tf.zeros([batch_size, self._num_writes, self._memory_size], dtype=link.dtype))

    def _precedence_weights(self, prev_precedence_weights, write_weights):
        """Calculates the new precedence weights given the current write weights.
        The precedence weights are the "aggregated write weights" for each write
        head, where write weights with sum close to zero will leave the precedence
        weights unchanged, but with sum close to one will replace the precedence weights.
        Args:
            prev_precedence_weights: `[batch_size, num_writes, memory_size]` 
            write_weights: `[batch_size, num_writes, memory_size]`
                the new write weights.
        Returns:
            new precedence weights `[batch_size, num_writes, memory_size]` 
        """
        with tf.name_scope('precedence_weights'):
            write_sum = tf.reduce_sum(write_weights, 2, keepdims=True)
            return (1 - write_sum) * prev_precedence_weights + write_weights

    
