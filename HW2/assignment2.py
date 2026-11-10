#################################
# Your name: Yoav Weinshall
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """


    def sample_from_D(self, m) -> np.ndarray:
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x = np.random.uniform(0, 1, m)
        x = np.sort(x)
        p_pos = np.where((x < 0.2) | ((0.4 <= x) & (x < 0.6)) | ((0.8 <= x) & (x < 1.0)), 0.8, 0.1)
        y = np.random.binomial(1, p_pos)
        return np.column_stack((x, y))


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two-dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """

        results = np.zeros((len(range(m_first, m_last + 1, step)), 3))
        for n in range(m_first, m_last + 1, step):
            empirical_errors = np.zeros(T)
            true_errors = np.zeros(T)
            for i in range(T):
                sample = self.sample_from_D(n)
                best_intervals, empirical_error = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
                true_error = self.calculate_true_error(best_intervals)
                empirical_errors[i] = empirical_error / n
                true_errors[i] = true_error
            results[(n - m_first) // step] = [n, np.average(empirical_errors), np.average(true_errors)]
        plt.plot(results[:, 0], results[:, 1], label='Empirical Error')
        plt.plot(results[:, 0], results[:, 2], label='True Error')
        plt.xlabel('Sample Size (m)')
        plt.ylabel('Error')
        plt.title(f'Empirical and True Errors for k={k}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
        return results[:, 1:3]

            

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               k_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        results = np.zeros((len(range(k_first, k_last + 1, step)), 3))
        sample = self.sample_from_D(m)
        for k in range(k_first, k_last + 1, step):
            best_intervals, empirical_error = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
            true_error = self.calculate_true_error(best_intervals)
            results[(k - k_first) // step] = [k, empirical_error / m, true_error]
        plt.plot(results[:, 0], results[:, 1], label='Empirical Error')
        plt.plot(results[:, 0], results[:, 2], label='True Error')
        plt.xlabel('Number of Intervals (k)')
        plt.ylabel('Error')
        plt.title(f'Empirical and True Errors for m={m}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
        best_k = results[np.argmin(results[:, 1]), 0]
        print(f"The best k value according to the ERM algorithm is: {best_k}")
        return best_k


    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        data = self.sample_from_D(m)
        np.random.shuffle(data)
        train_size = int(0.8 * m)
        train_data = data[:train_size]
        train_data = train_data[train_data[:, 0].argsort()]
        validation_data = data[train_size:]
        best_k = None
        best_validation_error = float('inf')
        for k in range(1, 11):
            best_intervals, _ = intervals.find_best_interval(train_data[:, 0], train_data[:, 1], k)
            validation_error = self.calculate_validation_error(best_intervals, validation_data)
            if validation_error < best_validation_error:
                best_validation_error = validation_error
                best_k = k
        print(f"The best k value according to the cross-validation algorithm is: {best_k}")
        return best_k
    
    #################################
    # Place for additional methods

    def order_intervals(self, intervals: np.ndarray) -> np.ndarray:
        """Orders the intervals in ascending order.
        Input: intervals - a list of tuples, where each tuple represents an interval.

        Returns: A list of tuples, where each tuple represents an interval, ordered in ascending order.
        """
        if len(intervals) == 0:
            return intervals
        intervals = sorted(intervals, key=lambda x: x[0])
        sorted_intervals = []
        for interval in intervals:
            l, u = interval
            if l < 0:
                l = 0
            if u > 1:
                u = 1
            if l > u:
                raise ValueError(f"Invalid interval: {interval}. The lower bound must be less than or equal to the upper bound.")
            sorted_intervals.append((l, u))
            
        merged_intervals = []
        current_l, current_u = sorted_intervals[0]
        for interval in sorted_intervals[1:]:
            if current_u >= interval[0]:  # Overlapping intervals
                current_u = max(current_u, interval[1])  # Merge intervals
            else:
                merged_intervals.append((current_l, current_u))
                current_l, current_u = interval
        merged_intervals.append((current_l, current_u))
        return np.array(merged_intervals)


    def calculate_true_error(self, intervals: np.ndarray) -> float:
        """Calculates the true error of a given hypothesis.
        Input: intervals - a list of tuples, where each tuple represents an interval.

        Returns: The true error (a float) of the hypothesis defined by the intervals.
        """
        intervals = self.order_intervals(intervals)
        positive_intervals = [[0.0, 0.2], [0.4, 0.6], [0.8, 1.0]]
        negative_intervals = [[0.2, 0.4], [0.6, 0.8]]

        #base case: if the interval is empty, the true error is p(y=1) = p(y=1|x in [0,1]) = 0.6 * 0.8 + 0.4 * 0.1
        true_error = 0.6 * 0.8 + 0.4 * 0.1

        for interval in intervals:
            l, u = interval

            # Calculates the reduction in true error due to the positive intervals
            for pos_interval in positive_intervals:
                pl, pu = pos_interval
                if u < pl or l > pu:  # No overlap
                    continue
                overlap_l = max(l, pl)
                overlap_u = min(u, pu)
                overlap_length = max(0, overlap_u - overlap_l)
                true_error -= 0.6 * overlap_length  # Subtract the positive contribution

            # Calculates the increase in true error due to the negative intervals
            for neg_interval in negative_intervals:
                nl, nu = neg_interval
                if u < nl or l > nu:  # No overlap
                    continue
                overlap_l = max(l, nl)
                overlap_u = min(u, nu)
                overlap_length = max(0, overlap_u - overlap_l)
                true_error += 0.8 * overlap_length  # Add the negative contribution

        return true_error


    def calculate_validation_error(self, intervals: np.ndarray, validation_data: np.ndarray) -> float:
        """Calculates the validation error of a given hypothesis on the validation data.
        Input: intervals - a list of tuples, where each tuple represents an interval.
               validation_data - a numpy array of shape (n, 2), where n is the number of validation samples.

        Returns: The validation error (a float) of the hypothesis defined by the intervals on the validation data.
        """
        intervals = self.order_intervals(intervals)
        error_count = 0
        for data_point in validation_data:
            x, y = data_point
            predicted_label = 0
            for interval in intervals:
                if interval[0] <= x <= interval[1]:
                    predicted_label = 1
                    break
            if predicted_label != y:
                error_count += 1
        return error_count / len(validation_data)
        


    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.cross_validation(1500)

