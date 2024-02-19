import mitsuba as mi
import drjit as dr
import torch
import math
from tools.mitsuba_tools import m2t

mi.set_variant("cuda_ad_rgb")


class MicrofacetDistribution:
    """
    Base class for microfacet distributions
    ref:
    https://github.com/mitsuba-renderer/mitsuba3/blob/master/include/mitsuba/render/microfacet.h
    """

    def D(self, wh):
        """ Microfacet distribution function

        Args:
            wh: microfacet normal

        Returns:
            D: (torch.Tensor)

        """

        pass

    def smith_g1(self, w, wh):
        """ Masking-shadowing function (Monodirectional shadowing function)
        which gives the fraction of microfacets with normal wh that are visible from direction w
        0<= G1 <= 1

        Args:
            w: view direction
            wh: microfacet normal

        Returns:
            G1

        """

        pass

    def G(self, wo, wi, wh):
        """ Bidirectional shadowing-masking function
        which gives the fraction of microfacets in a differential area that are visible from both directions wo and wi
        Smith's separable shadowing-masking approximation

        Args:
            wo: outgoing direction
            wi: incident direction
            wh: microfacet normal

        Returns:
            G

        """

        return self.smith_g1(wi, wh) * self.smith_g1(wo, wh)

    def Sample_wh(self, wo, sample):
        """ Sampling microfacet normal wh

        Args:
            wo: outgoing direction
            sample: uniform sample

        Returns:
            wh: microfacet normal
            pdf

        """

        pass

    def PDF(self, wo, wh):
        """ PDF of the microfacet normal wh

        Args:
            wo: outgoing direction
            wh: microfacet normal

        Returns:
            PDF

        """

        pass


class BeckmannDistribution(MicrofacetDistribution):
    """
    Beckmann microfacet distribution
    ref:
    https://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models#eq:beckmann-isotropic
    """
    #
    # def __init__(self, roughness_x: torch.Tensor, roughness_y=None):
    #     if roughness_y is None:
    #         roughness_y = roughness_x
    #
    #     # self.alpha_x = roughness_x
    #     # self.alpha_y = roughness_y
    #     self.roughness_x = roughness_x
    #     self.roughness_y = roughness_y
    #     self.alpha_x = self.RoughnessToAlpha(roughness_x)
    #     self.alpha_y = self.RoughnessToAlpha(roughness_y)
    #
    #
    #
    # def RoughnessToAlpha(self, roughness: torch.Tensor):
    #     roughness = torch.maximum(roughness, torch.tensor(1e-3))
    #     x = torch.log(roughness)
    #     return 1.62142 + 0.819955 * x + 0.1734 * x * x + 0.0171201 * x * x * x + 0.000640711 * x * x * x * x
    #
    # def D(self, wh: mi.Vector3f):
    #     alpha_xy = self.alpha_x * self.alpha_y
    #     cos_theta = mi.Frame3f.cos_theta(wh)
    #     cos_theta_2 = cos_theta * cos_theta
    #
    #     cos_theta = m2t(cos_theta).clamp_min(1e-4)
    #     cos_theta_2 = m2t(cos_theta_2).clamp_min(1e-4)
    #
    #
    #     result = torch.exp(-((m2t(wh.x) / self.alpha_x) ** 2 + (m2t(wh.y) / self.alpha_y) ** 2) / cos_theta_2) / (
    #             dr.pi * alpha_xy * (cos_theta_2) ** 2).clamp_min(1e-4)
    #
    #     # Prevent potential numerical issues in other stages of the model
    #     result[result * cos_theta < 1e-20] = 0
    #
    #     return result
    #
    # def smith_g1(self, w: mi.Vector3f, wh: mi.Vector3f):
    #     xy_alpha_2 = (self.alpha_x * m2t(w.x)) ** 2 + (self.alpha_y * m2t(w.y)) ** 2
    #     tan_theta_alpha_2 = xy_alpha_2 / (m2t(w.z).clamp_min(1e-4)) ** 2
    #
    #     a = torch.rsqrt(tan_theta_alpha_2)
    #     a_sqr = a * a
    #
    #     # Use a fast and accurate (<0.35% rel. error) rational approximation to the shadowing-masking function
    #     result = (3.535 * a + 2.181 * a_sqr) / (1 + 2.276 * a + 2.577 * a_sqr)
    #     result[a >= 1.6] = 1
    #
    #     # perpendicular incidence -- no shadowing/masking
    #     result[xy_alpha_2 == 0] = 1
    #
    #     # Ensure consistent orientation (can't see the back of the microfacet from the front and vice versa)
    #     cond = (dr.dot(w, wh) * mi.Frame3f.cos_theta(w) <= 0)
    #     cond = m2t(cond).bool()
    #     result[cond] = 0
    #
    #     result = torch.clamp(result, 0, 1)
    #     return result
    #
    # @torch.no_grad()
    # def Sample_wh(self, wo, sample):
    #     """
    #     Sample visible area of normals for Beckmann distribution (Visible normal sampling)
    #     ref:
    #     https://github.com/mitsuba-renderer/mitsuba3/blob/master/include/mitsuba/render/microfacet.h#L298
    #     https://github.com/mmp/pbrt-v3/blob/master/src/core/microfacet.cpp#L196
    #     """
    #
    #     m_alpha_u = mi.Float(self.alpha_x.flatten().float())
    #     m_alpha_v = mi.Float(self.alpha_y.flatten().float())
    #
    #     # Step 1: stretch wi
    #     wi_p = dr.normalize(mi.Vector3f(m_alpha_u * wo.x, m_alpha_v * wo.y, wo.z))
    #
    #     sin_phi, cos_phi = mi.Frame3f.sincos_phi(wi_p)
    #     cos_theta = mi.Frame3f.cos_theta(wi_p)
    #
    #     # Step 2: simulate P22_{wi}(slope.x, slope.y, 1, 1)
    #     slope = self.sample_visible_11(cos_theta, sample)
    #
    #     # Step 3: rotate & unstretch
    #     slope = mi.Vector2f(
    #         dr.fma(cos_phi, slope.x, -sin_phi * slope.y) * m_alpha_u,
    #         dr.fma(sin_phi, slope.x, cos_phi * slope.y) * m_alpha_v)
    #
    #     # Step 4: compute normal & PDF
    #     wh = dr.normalize(mi.Vector3f(-slope.x, -slope.y, 1))
    #
    #     pdf = self.PDF(wo, wh)
    #     return wh, pdf
    #
    # @torch.no_grad()
    # def PDF(self, wo: mi.Vector3f, wh: mi.Vector3f):
    #     pdf = self.D(wh) * self.smith_g1(wo, wh) * m2t(dr.abs_dot(wo, wh)) / m2t(mi.Frame3f.cos_theta(wo)).clamp_min(1e-4)
    #     return torch.clamp(pdf, 0, None)
    #
    # @torch.no_grad()
    # def sample_visible_11(self, cos_theta_i: mi.Float, sample: mi.Point2f):
    #     """brief Visible normal sampling code for the alpha=1 case (Beckmann distribution)
    #     ref:
    #     https://github.com/mitsuba-renderer/mitsuba3/blob/master/include/mitsuba/render/microfacet.h#L368
    #     """
    #
    #     tan_theta_i = dr.safe_sqrt(dr.fma(cos_theta_i, -cos_theta_i, 1.)) / cos_theta_i
    #     cot_theta_i = dr.rcp(tan_theta_i)
    #
    #     maxval = dr.erf(cot_theta_i)
    #
    #     sample = dr.maximum(dr.minimum(sample, 1. - 1e-6), 1e-6)
    #     x = maxval - (maxval + 1.) * dr.erf(dr.sqrt(-dr.log(sample.x)))
    #
    #     sample.x *= 1. + maxval + dr.inv_sqrt_pi * tan_theta_i * dr.exp(-dr.sqr(cot_theta_i))
    #
    #     # Three Newton iterations
    #     for i in range(3):
    #         slope = dr.erfinv(x)
    #         value = 1. + x + dr.inv_sqrt_pi * tan_theta_i * dr.exp(-dr.sqr(slope)) - sample.x
    #         derivative = 1. - slope * tan_theta_i
    #         x -= value / derivative
    #
    #
    #     return dr.erfinv(mi.Vector2f(x, dr.fma(2., sample.y, -1.)))


class TrowbridgeReitzDistribution(MicrofacetDistribution):
    """
    TrowbridgeReitz microfacet distribution
    ref:
    https://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models#eq:beckmann-isotropic
    """

    def __init__(self, roughness_x, roughness_y=None):
        if roughness_y is None:
            roughness_y = roughness_x
        # self.roughness_x = roughness_x
        # self.roughness_y = roughness_y
        # self.alpha_x = self.RoughnessToAlpha(roughness_x)
        # self.alpha_y = self.RoughnessToAlpha(roughness_y)
        self.alpha_x = roughness_x
        self.alpha_y = roughness_y

    # def RoughnessToAlpha(self, roughness):
    #     roughness = torch.maximum(roughness, torch.tensor(1e-3))
    #     x = torch.log(roughness)
    #     return 1.62142 + 0.819955 * x + 0.1734 * x * x + 0.0171201 * x * x * x + 0.000640711 * x * x * x * x

    def D(self, wh):
        alpha_xy = self.alpha_x * self.alpha_y
        cos_theta = mi.Frame3f.cos_theta(wh)

        cos_theta = m2t(cos_theta)
        result = 1 / (dr.pi * alpha_xy * (
                    (m2t(wh.x) / self.alpha_x) ** 2 + (m2t(wh.y) / self.alpha_y) ** 2 + m2t(wh.z) ** 2) ** 2).clamp_min(1e-4)

        # Prevent potential numerical issues in other stages of the model
        result[result * cos_theta < 1e-20] = 0

        return result

    def smith_g1(self, w, wh):
        xy_alpha_2 = (self.alpha_x * m2t(w.x)) ** 2 + (self.alpha_y * m2t(w.y)) ** 2
        tan_theta_alpha_2 = xy_alpha_2 / ((m2t(w.z)) ** 2).clamp_min(1e-4)

        result = 2. / (1. + torch.sqrt(1. + tan_theta_alpha_2))


        # perpendicular incidence -- no shadowing/masking
        result[xy_alpha_2 == 0] = 1

        # Ensure consistent orientation (can't see the back of the microfacet from the front and vice versa)
        cond = (dr.dot(w, wh) * mi.Frame3f.cos_theta(w) <= 0)
        cond = m2t(cond).bool()
        result[cond] = 0

        result = torch.clamp(result, 0, 1)

        return result

    @torch.no_grad()
    def Sample_wh(self, wo, sample):
        """
        Sample visible area of normals for TrowbridgeReitz distribution (Visible normal sampling)
        ref:
        https://github.com/mitsuba-renderer/mitsuba3/blob/master/include/mitsuba/render/microfacet.h#L298
        https://github.com/mmp/pbrt-v3/blob/master/src/core/microfacet.cpp#L196
        """

        m_alpha_u = mi.Float(self.alpha_x.flatten().float())
        m_alpha_v = mi.Float(self.alpha_y.flatten().float())

        # Step 1: stretch wi
        wi_p = dr.normalize(mi.Vector3f(m_alpha_u * wo.x, m_alpha_v * wo.y, wo.z))

        sin_phi, cos_phi = mi.Frame3f.sincos_phi(wi_p)
        cos_theta = mi.Frame3f.cos_theta(wi_p)

        # Step 2: simulate P22_{wi}(slope.x, slope.y, 1, 1)
        slope = self.sample_visible_11(cos_theta, sample)

        # Step 3: rotate & unstretch
        slope = mi.Vector2f(
            dr.fma(cos_phi, slope.x, -sin_phi * slope.y) * m_alpha_u,
            dr.fma(sin_phi, slope.x, cos_phi * slope.y) * m_alpha_v)

        # Step 4: compute normal & PDF
        wh = dr.normalize(mi.Vector3f(-slope.x, -slope.y, 1))

        pdf = self.PDF(wo, wh)
        return wh, pdf

    @torch.no_grad()
    def PDF(self, wo: mi.Vector3f, wh: mi.Vector3f):
        pdf = self.D(wh) * self.smith_g1(wo, wh) * m2t(dr.abs_dot(wo, wh)) / m2t(mi.Frame3f.cos_theta(wo)).clamp_min(1e-4)
        return torch.clamp(pdf, 0, None)

    @torch.no_grad()
    def sample_visible_11(self, cos_theta_i, sample):
        """brief Visible normal sampling code for the alpha=1 case (TrowbridgeReitz distribution)
        ref:
        https://github.com/mitsuba-renderer/mitsuba3/blob/master/include/mitsuba/render/microfacet.h#L368
        """
        p = mi.warp.square_to_uniform_disk_concentric(sample)

        s = 0.5 * (1. + cos_theta_i)
        p.y = dr.lerp(dr.safe_sqrt(1. - dr.sqr(p.x)), p.y, s)

        x = p.x
        y = p.y
        z = dr.safe_sqrt(1. - dr.squared_norm(p))

        sin_theta_i = dr.safe_sqrt(1. - dr.sqr(cos_theta_i))
        norm = dr.rcp(dr.fma(sin_theta_i, y, cos_theta_i * z))

        return mi.Vector2f(dr.fma(cos_theta_i, y, -sin_theta_i * z), x) * norm
