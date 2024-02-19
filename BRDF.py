import mitsuba as mi
import drjit as dr
import torch
from tools.mitsuba_tools import m2t
from MicrofacetDistribution import BeckmannDistribution, TrowbridgeReitzDistribution
import math

mi.set_variant("cuda_ad_rgb")


class BRDF:
    """
    Base class for BRDFs
    """

    def f(self, wo, wi, active):
        """ Evaluate the BRDF

        Args:
            wo: outgoing ray
            wi: incident ray

        Returns:
            f: (torch.Tensor [N, 3]) the value of the distribution function for the given pair of directions

        """

        pass

    def sample_f(self, wo, sample, active):
        """ Sample the BRDF and return the sampled direction, the value of the distribution function, and the corresponding PDF

        Args:
            wo: outgoing ray
            sample: a sample from a uniform distribution

        Returns:
            wi: (mi.Vector3f) the sampled (incident) direction
            pdf: (torch.Tensor [N, 3]) the corresponding PDF
            f: (torch.Tensor [N, 3]) the value of the distribution function


        """

        pass

    def PDF(self, wo, wi, active):
        """  Evaluate the PDF for the given pair of directions

        Args:
            wo: outgoing ray
            wi: incident ray

        Returns:
            pdf (torch.Tensor [N, 3]): the corresponding PDF

        """

        pass

    def eval_pdf(self, wo, wi, active):
        """ use for evaluate wi sampled from Emitter

        Args:
            wo:
            wi:
            active:

        Returns:
            value: f(wo,wi) * cos_term
            pdf:

        """
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        cos_theta_i = mi.Frame3f.cos_theta(wi)
        active = active & (cos_theta_i > 0) & (cos_theta_o > 0)
        value = self.f(wo, wi, active) * m2t(cos_theta_i)
        pdf = self.PDF(wo, wi, active)

        # print(f"value: {value.max().item():.2f}, {value.min().item():.2f}")
        # print(f"pdf: {pdf.max().item():.2f}, {pdf.min().item():.2f}")
        # print('*'*20)

        value = torch.where(m2t(active).bool(), value, torch.zeros_like(value))
        return value, pdf

    def sample(self, wo, sample, sample1d, active):
        """ use for sample wi for path tracing

        Args:
            wo:
            wi:
            sample:
            active:

        Returns:
            wi:
            pdf
            value: f(wo,wi) * cos_term / pdf

        """
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        active = active & (cos_theta_o > 0)

        wi, pdf, f = self.sample_f(wo, sample, sample1d, active)

        cos_theta_i = mi.Frame3f.cos_theta(wi)
        active = active & (cos_theta_i > 0)

        inv_pdf = 1 / pdf.clamp_min(1e-4)
        inv_pdf[pdf == 0] = 0
        value = f * inv_pdf * m2t(cos_theta_i)
        value = torch.where(m2t(active).bool(), value, torch.zeros_like(value))
        value[value.isnan()] = 0

        return wi, pdf, value




class LambertianReflection(BRDF):
    """
    Lambertian BRDF
    """

    def __init__(self, Rd: torch.Tensor):
        self.Rd = Rd  # roughness

    def f(self, wo: mi.Vector3f, wi: mi.Vector3f, active=mi.Bool(True)):
        f = self.Rd * dr.inv_pi

        cos_theta_i = mi.Frame3f.cos_theta(wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        active = active & (cos_theta_i > 0) & (cos_theta_o > 0)
        f = torch.where(m2t(active).bool(), f, torch.zeros_like(f))

        f[f.isnan()] = 0
        f = f.clamp_min(0)
        return f

    def sample_f(self, wo: mi.Vector3f, sample: mi.Point2f, sample1d: mi.Point1f, active=mi.Bool(True)):
        wi = mi.warp.square_to_cosine_hemisphere(sample)
        pdf = self.PDF(wo, wi, active)

        f = self.f(wo, wi, active)

        return wi, pdf, f

    @torch.no_grad()
    def PDF(self, wo: mi.Vector3f, wi: mi.Vector3f, active=mi.Bool(True)):
        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wi)
        pdf = pdf.torch().unsqueeze(1)

        cos_theta_i = mi.Frame3f.cos_theta(wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        active = active & (cos_theta_i > 0) & (cos_theta_o > 0)
        pdf = torch.where(m2t(active).bool(), pdf, torch.zeros_like(pdf))

        pdf[pdf.isnan()] = 0
        pdf = pdf.clamp_min(0)
        return pdf


class MicrofacetReflection(BRDF):
    """
    Torrance–Sparrow microfacet model
    """

    def __init__(self, Rs, roughness_x, roughness_y=None, distr_type='Beckmann'):
        self.Rs = Rs  # reflectance of the surface at normal incidence
        # if roughness_y is None:
        #     roughness_y = roughness_x

        assert distr_type in ['Beckmann', 'GGX']
        if distr_type == 'Beckmann':
            # self.distribution = BeckmannDistribution(roughness_x, roughness_y)
            self.distribution = BeckmannDistribution(roughness_x)
        elif distr_type == 'GGX':
            # self.distribution = TrowbridgeReitzDistribution(roughness_x, roughness_y)
            self.distribution = TrowbridgeReitzDistribution(roughness_x)

    # Schlick approximation for Fresnel reflectance
    def Fresnel_Schlick(self, cosTheta):
        Rs = self.Rs
        F = Rs + (1 - Rs) * (1 - cosTheta) ** 5
        return F

    def f(self, wo, wi, active):
        wh = dr.normalize(wo + wi)
        F = self.Fresnel_Schlick(m2t(dr.dot(wi, wh)))
        D = self.distribution.D(wh)
        G = self.distribution.G(wo, wi, wh)

        # print(f"F: {F.max().item():.2f}, {F.min().item():.2f}")
        # print(f"D: {D.max().item():.2f}, {D.min().item():.2f}")
        # print(f"G: {G.max().item():.2f}, {G.min().item():.2f}")

        cos_theta_o = mi.Frame3f.cos_theta(wo)
        cos_theta_i = mi.Frame3f.cos_theta(wi)

        f = F * D * G

        f = f / (4 * m2t(cos_theta_o) * m2t(cos_theta_i)).clamp_min(1e-4)  # clip

        active = active & (cos_theta_i > 0) & (cos_theta_o > 0)
        f[D == 0.0] = 0.0  # wh=[0,0,0]

        f = torch.where(m2t(active).bool(), f, torch.zeros_like(f))

        f[f.isnan()] = 0
        f = f.clamp_min(0)

        return f

    def sample_f(self, wo, sample, sample1d, active):
        wh, pdf = self.distribution.Sample_wh(wo, sample)
        wi = mi.reflect(wo, wh)

        pdf = torch.where(m2t(active).bool(), pdf, torch.zeros_like(pdf))
        pdf = pdf / (4 * m2t(dr.dot(wo, wh))).clamp_min(1e-4)

        f = self.f(wo, wi, active)

        # Ensure that this is a valid sample
        f[pdf == 0] = 0
        f[m2t(mi.Frame3f.cos_theta(wi) <= 0).bool()] = 0

        return wi, pdf, f

    @torch.no_grad()
    def PDF(self, wo, wi, active):
        wh = dr.normalize(wo + wi)
        pdf = self.distribution.PDF(wo, wh) / (4 * m2t(dr.dot(wo, wh)).clamp_min(1e-4))

        cos_theta_i = mi.Frame3f.cos_theta(wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        cos_theta_ih = dr.dot(wi, wh)
        cos_theta_oh = dr.dot(wo, wh)

        active = active & (cos_theta_i > 0) & (cos_theta_o > 0) & (cos_theta_ih > 0) & (cos_theta_oh > 0)

        pdf = torch.where(m2t(active).bool(), pdf, torch.zeros_like(pdf))

        pdf[pdf.isnan()] = 0
        pdf = pdf.clamp_min(0)

        return pdf


class Phong(BRDF):
    def __init__(self, Rs: torch.Tensor, n: torch.Tensor):
        self.Rs = Rs
        self.n = n * 200

    def f(self, wo, wi, active):

        n = self.n
        alpha = dr.dot(wi, mi.reflect(wo)).torch()[:, None]
        alpha = torch.clip(alpha, 0, None)
        # alpha = torch.nan_to_num(alpha, nan=0, posinf=0, neginf=0)  # wi can be nan

        f = (n + 2) * (1 / (2 * dr.pi)) * torch.pow(alpha, n) * self.Rs

        cos_theta_i = mi.Frame3f.cos_theta(wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        active = active & (cos_theta_i > 0) & (cos_theta_o > 0)
        f = torch.where(m2t(active).bool(), f, torch.zeros_like(f))

        return f


    def sample_f(self, wo, sample, sample1d, active):
        n = self.n
        n_mi = mi.Float(n.flatten().float())

        R = mi.reflect(wo)

        sinAlpha = dr.sqrt(1 - dr.power(sample.y, 2 / (n_mi + 1)))
        cosAlpha = dr.power(sample.y, 1 / (n_mi + 1))
        phi = (2.0 * dr.pi) * sample.x
        localDir = mi.Vector3f(sinAlpha * dr.cos(phi),
                               sinAlpha * dr.sin(phi),
                               cosAlpha)
        wi = mi.Frame3f(R).to_world(localDir)

        pdf = self.PDF(wo, wi, active)
        f = self.f(wo, wi, active)

        return wi, pdf, f


    @torch.no_grad()
    def PDF(self, wo, wi, active):
        n = self.n
        alpha = dr.dot(wi, mi.reflect(wo)).torch()[:, None]
        alpha = torch.clip(alpha, 0, None)
        # alpha = torch.nan_to_num(alpha, nan=0, posinf=0, neginf=0)  # wi can be nan

        pdf = (n + 1) * (1 / (2 * dr.pi)) * torch.pow(alpha, n)

        cos_theta_i = mi.Frame3f.cos_theta(wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        active = active & (cos_theta_i > 0) & (cos_theta_o > 0)
        pdf = torch.where(m2t(active).bool(), pdf, torch.zeros_like(pdf))

        return pdf




class GlossyBRDF(BRDF):
    """
    mixed BRDF
    """

    def __init__(self, Rd: torch.Tensor, Rs, roughness_x, roughness_y=None, specular_type='GGX'):

        assert specular_type in ['GGX', 'Phong']
        self.LambertianReflection = LambertianReflection(Rd)

        # roughness_x in [0.02, 1]
        if specular_type == 'GGX':
            self.SpecularReflection = MicrofacetReflection(Rs, roughness_x, roughness_y, specular_type)
        else:
            self.SpecularReflection = Phong(Rs, (5 / roughness_x))  # n = 4 / roughness, in [1, 247]
        self.ratio = 0.5

    def f(self, wo, wi, active):
        f = self.LambertianReflection.f(wo, wi, active) + self.SpecularReflection.f(wo, wi, active)
        return f

    def sample_f(self, wo, sample, sample1d, active=mi.Bool(True)):
        wi, _, _ = self.LambertianReflection.sample_f(wo, sample, sample1d, active)
        wi2, _, _ = self.SpecularReflection.sample_f(wo, sample, sample1d, active)

        wi = dr.select(sample1d < self.ratio, wi2, wi)

        pdf = self.PDF(wo, wi, active)
        f = self.f(wo, wi, active)


        return wi, pdf, f

    @torch.no_grad()
    def PDF(self, wo, wi, active=mi.Bool(True)):
        pdf = (1 - self.ratio) * self.LambertianReflection.PDF(wo, wi, active) + self.ratio * self.SpecularReflection.PDF(wo, wi, active)
        return pdf




class GlossyBRDF_cws(BRDF):
    """
    mixed BRDF
    """

    def __init__(self, Rd: torch.Tensor, Rs, roughness_x, roughness_y=None, specular_type='GGX'):

        assert specular_type in ['GGX', 'Phong']
        self.LambertianReflection = LambertianReflection(Rd)

        # roughness_x in [0.02, 1]
        if specular_type == 'GGX':
            self.SpecularReflection = MicrofacetReflection(Rs, roughness_x, roughness_y, specular_type)
        else:
            self.SpecularReflection = Phong(Rs, (5 / roughness_x))  # n = 4 / roughness, in [1, 247]
        self.ratio = 0.5

    def f(self, wo, wi, active):
        f = self.LambertianReflection.f(wo, wi, active) + self.SpecularReflection.f(wo, wi, active)
        return f

    def sample_f(self, wo, sample, sample1d, active=mi.Bool(True)):
        wi = mi.warp.square_to_cosine_hemisphere(sample)
        f = self.f(wo, wi, active)
        pdf = self.PDF(wo, wi, active)
        return wi, pdf, f

    @torch.no_grad()
    def PDF(self, wo, wi, active=mi.Bool(True)):
        pdf = mi.warp.square_to_cosine_hemisphere_pdf(wi)
        pdf = pdf.torch().unsqueeze(1)

        cos_theta_i = mi.Frame3f.cos_theta(wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        active = active & (cos_theta_i > 0) & (cos_theta_o > 0)
        pdf = torch.where(m2t(active).bool(), pdf, torch.zeros_like(pdf))

        pdf[pdf.isnan()] = 0
        pdf = pdf.clamp_min(0)
        return pdf



class GlossyBRDF_random(BRDF):
    """
    mixed BRDF
    """

    def __init__(self, Rd: torch.Tensor, Rs, roughness_x, roughness_y=None, specular_type='GGX'):

        assert specular_type in ['GGX', 'Phong']
        self.LambertianReflection = LambertianReflection(Rd)

        # roughness_x in [0.02, 1]
        if specular_type == 'GGX':
            self.SpecularReflection = MicrofacetReflection(Rs, roughness_x, roughness_y, specular_type)
        else:
            self.SpecularReflection = Phong(Rs, (5 / roughness_x))  # n = 4 / roughness, in [1, 247]
        self.ratio = 0.5

    def f(self, wo, wi, active):
        f = self.LambertianReflection.f(wo, wi, active) + self.SpecularReflection.f(wo, wi, active)
        return f

    def sample_f(self, wo, sample, sample1d, active=mi.Bool(True)):
        wi = mi.warp.square_to_uniform_hemisphere(sample)
        f = self.f(wo, wi, active)
        pdf = self.PDF(wo, wi, active)

        return wi, pdf, f

    @torch.no_grad()
    def PDF(self, wo, wi, active=mi.Bool(True)):
        pdf = mi.warp.square_to_uniform_hemisphere_pdf(wi)
        pdf = pdf.torch().unsqueeze(1)

        cos_theta_i = mi.Frame3f.cos_theta(wi)
        cos_theta_o = mi.Frame3f.cos_theta(wo)
        active = active & (cos_theta_i > 0) & (cos_theta_o > 0)
        pdf = torch.where(m2t(active).bool(), pdf, torch.zeros_like(pdf))

        pdf[pdf.isnan()] = 0
        pdf = pdf.clamp_min(0)

        return pdf