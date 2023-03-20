# -*- coding: utf-8 -*-
import numpy as np
import math


class Butterlife():
    """ Butterworth filter class """

    def __init__(self):
        pass

    def check_params(self, order):
        """ Check parameters """

        assert order < 5, "Filter order is too high (order max = 4, given = "\
            + str(order) + ")"

    def assign_params(self, sig, fs, fc, order, ftype, padding):
        """ Assign parameters """

        self.sig_ = sig
        self.fs_ = fs
        self.fc_ = fc
        self.order_ = order
        self.ftype_ = ftype
        self.padding_ = padding
        self.a_ = np.zeros(order+1)
        self.b_ = np.zeros(order+1)
        self.c_ = 1 / math.tan(fc * math.pi / fs)

        # ---- Get butterworth coefficients
        self.get_coefficients()

    def filt(self, sig, fs, fc, order, ftype, padding=True):
        """ Filter butterworth : low and high pass filter

        Parameters
        ------------------------------------------
        sig: input signal
        fs: Sampling frequency Hz
        fc: cut off frequency
        ftype: Filter type ("low "or "high")
        padding: Flag for keeping original signal size

        Returns
        ------------------------------------------
        filtered signal

        """

        # Chek params
        self.check_params(order)

        # Assign params
        self.assign_params(sig, fs, fc, order, ftype, padding)

        # ---- Get butterworth coefficients
        a = self.a_
        b = self.b_

        # ---- Set params for filtering
        n = len(sig)
        # Extend signal size to init filtering
        sig_padding = np.zeros((n + order + 1, ))
        for i in range(n):
            sig_padding[i + order] = sig[i]
        sig_padding[:order] = sig[0]
        sig_padding[(n-1):] = sig[-1]

        # ---- Y: 1st filter step
        sig_filt1 = np.zeros(n + order + 1)
        sig_filt1[:order] = sig[0]

        for i in range(order, n + 1):
            y = 0
            for k in range(len(b)):
                y += b[k]*sig_padding[i-k] - a[k]*sig_filt1[i-k]
            sig_filt1[i] = y
        sig_filt1[n:] = sig_filt1[n]

        # ---- Z: 2nd filter step
        sig_filt2 = np.zeros(n + 1)
        for i in range(order):
            sig_filt2[n-i] = sig_filt1[n + order - i]

        for i in range(n - order, -1, -1):
            z = 0
            for k in range(len(b)):
                z += b[k]*sig_filt1[i+k+order] - a[k]*sig_filt2[i+k]
            sig_filt2[i] = z

        # ---- Remove dephasing
        sig_filt = np.zeros(n - 1)
        for i in range(len(sig_filt)):
            sig_filt[i] = sig_filt2[i+1]

        # Add sample to keep original signal size
        if padding:
            padding_sample = sig_filt[0]
            sig_filt_final = []
            sig_filt_final.append(padding_sample)
            sig_filt_final.extend(sig_filt)
            sig_filt_final = np.array(sig_filt_final)

        self.sig_filt_ = sig_filt_final

        return sig_filt_final

    def get_coefficients(self):
        """ get butterworth filter coeffcients """

        ftype = self.ftype_

        self.get_lowpass_coefficients()

        if ftype == 'high':
            self.get_highpass_coefficients()

    def get_lowpass_coefficients(self):
        """ get butterworth lowpass filter coeffcients """
        order = self.order_
        a = self.a_
        b = self.b_
        c = self.c_

        if order == 1:
            d = c + 1

            # Coefs a
            a[0] = 1
            a[1] = (1 - c)/d

            # Coefs b
            b[0] = 1/d
            b[1] = b[0]

        elif order == 2:
            q0 = np.sqrt(2)  # resonance term
            d = c**2 + q0*c + 1

            # Coefs a
            a[0] = 1
            a[1] = (- 2*c**2 + 0*c + 2)/d
            a[2] = (1*c**2 - q0*c + 1)/d

            # Coefs b
            b[0] = 1/d
            b[1] = 2*b[0]
            b[2] = 1*b[0]

        elif order == 3:
            d = c**3 + 2*c**2 + 2*c + 1

            # Coefs a
            a[0] = 1
            a[1] = (- 3*c**3 - 2*c**2 + 2*c + 3)/d
            a[2] = (3*c**3 - 2*c**2 - 2*c + 3)/d
            a[3] = (- 1*c**3 + 2*c**2 - 2*c + 1)/d

            # Coefs b
            b[0] = 1/d
            b[1] = 3*b[0]
            b[2] = 3*b[0]
            b[3] = 1*b[0]

        elif order == 4:
            q0 = 0.7654
            q1 = 1.8478
            e = (q0 + q1)*c**3
            f = (2 + q0*q1)*c**2
            g = (q0 + q1)*c

            d = c**4 + e + f + g + 1

            # Coefs a
            a[0] = 1
            a[1] = (- 4*c**4 - 2*e + 0*f + 2*g + 4)/d
            a[2] = (6*c**4 + 0*e - 2*f + 0*g + 6)/d
            a[3] = (- 4*c**4 + 2*e + 0*f - 2*g + 4)/d
            a[4] = (1*c**4 - 1*e + 1*f - 1*g + 1)/d

            # Coefs b
            b[0] = 1/d
            b[1] = 4*b[0]
            b[2] = 6*b[0]
            b[3] = 4*b[0]
            b[4] = 1*b[0]

        self.a_ = a
        self.b_ = b

    def get_highpass_coefficients(self):
        """ get butterworth highpass filter coeffcients """

        order = self.order_
        b = self.b_
        c = self.c_

        if order == 1:
            b[0] = b[0] * (-c)**order
            b[1] = - b[1] * (-c)**order

        elif order == 2:
            b[0] = b[0] * (-c)**order
            b[1] = - b[1] * (-c)**order
            b[2] = b[2] * (-c)**order

        elif order == 3:
            b[0] = b[0] * (-c)**order
            b[1] = - b[1] * (-c)**order
            b[2] = b[2] * (-c)**order
            b[3] = - b[3] * (-c)**order

        elif order == 4:
            b[0] = b[0] * (-c)**order
            b[1] = - b[1] * (-c)**order
            b[2] = b[2] * (-c)**order
            b[3] = - b[3] * (-c)**order
            b[4] = b[4] * (-c)**order

        self.b_ = b
