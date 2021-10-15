#pragma once
#include "host_tensor.hpp"

template <typename T>
inline auto activ(T v, const ck::index_t activ_type)
{
    const T alpha = 0.30000001192092896;
    switch(activ_type)
    {
    case 0: return v;
    case 1: return (v >= 0 ? v : alpha * v);
    case 2: return (1 / (1 + exp(-v)));
    default: throw std::runtime_error("unsupported activ type"); break;
    }
}

template <typename TIn,
          typename TWei,
          typename TOut,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void host_direct_convolution(const Tensor<TIn>& in,
                             const Tensor<TWei>& wei,
                             Tensor<TOut>& out,
                             const ConvStrides& conv_strides,
                             const ConvDilations& conv_dilations,
                             const InLeftPads& in_left_pads,
                             const InRightPads&,
                             const ConvTensorLayout layout = ConvTensorLayout::NCHW,
                             const ck::index_t activ_type  = 0)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    auto f_nchw = [&](auto n, auto k, auto ho, auto wo) {
        double v = 0;
        for(int c = 0; c < wei.mDesc.GetLengths()[1]; ++c)
        {
            for(int y = 0; y < wei.mDesc.GetLengths()[2]; ++y)
            {
                int hi = ho * conv_strides[I0] + y * conv_dilations[I0] - in_left_pads[I0];
                for(int x = 0; x < wei.mDesc.GetLengths()[3]; ++x)
                {
                    int wi = wo * conv_strides[I1] + x * conv_dilations[I1] - in_left_pads[I1];
                    if(hi >= 0 && hi < in.mDesc.GetLengths()[2] && wi >= 0 &&
                       wi < in.mDesc.GetLengths()[3])
                    {
                        v += static_cast<const double>(in(n, c, hi, wi)) *
                             static_cast<const double>(wei(k, c, y, x));
                    }
                }
            }
        }
        out(n, k, ho, wo) = activ(v, activ_type);
    };

    auto f_nhwc = [&](auto n, auto ho, auto wo, auto k) {
        double v = 0;
        for(int c = 0; c < wei.mDesc.GetLengths()[3]; ++c)
        {
            for(int y = 0; y < wei.mDesc.GetLengths()[1]; ++y)
            {
                int hi = ho * conv_strides[I0] + y * conv_dilations[I0] - in_left_pads[I0];
                for(int x = 0; x < wei.mDesc.GetLengths()[2]; ++x)
                {
                    int wi = wo * conv_strides[I1] + x * conv_dilations[I1] - in_left_pads[I1];
                    if(hi >= 0 && hi < in.mDesc.GetLengths()[1] && wi >= 0 &&
                       wi < in.mDesc.GetLengths()[2])
                    {
                        v += static_cast<const double>(in(n, hi, wi, c)) *
                             static_cast<const double>(wei(k, y, x, c));
                    }
                }
            }
        }
        out(n, ho, wo, k) = activ(v, activ_type);
    };

    if(layout == ConvTensorLayout::NCHW)
    {
        make_ParallelTensorFunctor(f_nchw,
                                   out.mDesc.GetLengths()[0],
                                   out.mDesc.GetLengths()[1],
                                   out.mDesc.GetLengths()[2],
                                   out.mDesc.GetLengths()[3])(std::thread::hardware_concurrency());
    }
    else if(layout == ConvTensorLayout::NHWC)
    {
        make_ParallelTensorFunctor(f_nhwc,
                                   out.mDesc.GetLengths()[0],
                                   out.mDesc.GetLengths()[1],
                                   out.mDesc.GetLengths()[2],
                                   out.mDesc.GetLengths()[3])(std::thread::hardware_concurrency());
    }
    else
    {
        throw std::runtime_error("wrong! not supported layout");
    }
}

template <typename TIn,
          typename TWei,
          typename TOut,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void host_direct_convolution_nchwc(const Tensor<TIn>& in,
                                   const Tensor<TWei>& wei,
                                   const Tensor<TOut>& bias,
                                   Tensor<TOut>& out,
                                   const ConvStrides& conv_strides,
                                   const ConvDilations& conv_dilations,
                                   const InLeftPads& in_left_pads,
                                   const InRightPads&,
                                   const ck::index_t activ_type = 0)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    auto f_nchw = [&](auto n, auto k0, auto ho, auto wo, auto k1) {
        double v    = 0;
        const int k = k0 * out.mDesc.GetLengths()[4] + k1;
        for(int c0 = 0; c0 < wei.mDesc.GetLengths()[1]; ++c0)
        {
            for(int c1 = 0; c1 < wei.mDesc.GetLengths()[4]; ++c1)
            {
                for(int y = 0; y < wei.mDesc.GetLengths()[2]; ++y)
                {
                    int hi = ho * conv_strides[I0] + y * conv_dilations[I0] - in_left_pads[I0];
                    for(int x = 0; x < wei.mDesc.GetLengths()[3]; ++x)
                    {
                        int wi = wo * conv_strides[I1] + x * conv_dilations[I1] - in_left_pads[I1];
                        if(hi >= 0 && hi < in.mDesc.GetLengths()[2] && wi >= 0 &&
                           wi < in.mDesc.GetLengths()[3])
                        {
                            v += static_cast<const double>(in(n, c0, hi, wi, c1)) *
                                 static_cast<const double>(wei(k, c0, y, x, c1));
                        }
                    }
                }
            }
        }
        v += bias(k0, k1);
        out(n, k0, ho, wo, k1) = activ(v, activ_type);
    };

    make_ParallelTensorFunctor(f_nchw,
                               out.mDesc.GetLengths()[0],
                               out.mDesc.GetLengths()[1],
                               out.mDesc.GetLengths()[2],
                               out.mDesc.GetLengths()[3],
                               out.mDesc.GetLengths()[4])(std::thread::hardware_concurrency());
}

template <typename TIn,
          typename TWei,
          typename TOut,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void host_direct_convolution_add_nchwc(const Tensor<TIn>& in,
                                       const Tensor<TWei>& wei,
                                       const Tensor<TOut>& add,
                                       const Tensor<TOut>& bias,
                                       Tensor<TOut>& add_host,
                                       Tensor<TOut>& out_host,
                                       const ConvStrides& conv_strides,
                                       const ConvDilations& conv_dilations,
                                       const InLeftPads& in_left_pads,
                                       const InRightPads&,
                                       const ck::index_t activ_type = 0)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    auto f_nchw = [&](auto n, auto k0, auto ho, auto wo, auto k1) {
        double v = 0;
        for(int c0 = 0; c0 < wei.mDesc.GetLengths()[1]; ++c0)
        {
            for(int c1 = 0; c1 < wei.mDesc.GetLengths()[4]; ++c1)
            {
                for(int y = 0; y < wei.mDesc.GetLengths()[2]; ++y)
                {
                    int hi = ho * conv_strides[I0] + y * conv_dilations[I0] - in_left_pads[I0];
                    for(int x = 0; x < wei.mDesc.GetLengths()[3]; ++x)
                    {
                        int wi = wo * conv_strides[I1] + x * conv_dilations[I1] - in_left_pads[I1];
                        if(hi >= 0 && hi < in.mDesc.GetLengths()[2] && wi >= 0 &&
                           wi < in.mDesc.GetLengths()[3])
                        {
                            v += static_cast<const double>(in(n, c0, hi, wi, c1)) *
                                 static_cast<const double>(
                                     wei(k0 * out_host.mDesc.GetLengths()[4] + k1, c0, y, x, c1));
                        }
                    }
                }
            }
        }

        v = activ(v, activ_type) + bias(k0, k1);

        const int hox2 = ho * 2;
        const int wox2 = wo * 2;

        out_host(n, k0, ho, wo, k1) = v;

        add_host(n, k0, hox2, wox2, k1)         = v + add(n, k0, hox2, wox2, k1);
        add_host(n, k0, hox2, wox2 + 1, k1)     = v + add(n, k0, hox2, wox2 + 1, k1);
        add_host(n, k0, hox2 + 1, wox2, k1)     = v + add(n, k0, hox2 + 1, wox2, k1);
        add_host(n, k0, hox2 + 1, wox2 + 1, k1) = v + add(n, k0, hox2 + 1, wox2 + 1, k1);
    };

    make_ParallelTensorFunctor(f_nchw,
                               out_host.mDesc.GetLengths()[0],
                               out_host.mDesc.GetLengths()[1],
                               out_host.mDesc.GetLengths()[2],
                               out_host.mDesc.GetLengths()[3],
                               out_host.mDesc.GetLengths()[4])(std::thread::hardware_concurrency());
}

template <typename TIn,
          typename TWei,
          typename TOut,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void host_direct_convolution_maxpool_nchwc(const Tensor<TIn>& in,
                                           const Tensor<TWei>& wei,
                                           const Tensor<TOut>& bias,
                                           Tensor<TOut>& out_host,
                                           Tensor<TOut>& max_host,
                                           const ConvStrides& conv_strides,
                                           const ConvDilations& conv_dilations,
                                           const InLeftPads& in_left_pads,
                                           const InRightPads&,
                                           const ck::index_t activ_type = 0)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    auto f_nchw = [&](auto n, auto k0, auto ho, auto wo, auto k1) {
        double v = 0;
        for(int c0 = 0; c0 < wei.mDesc.GetLengths()[1]; ++c0)
        {
            for(int c1 = 0; c1 < wei.mDesc.GetLengths()[4]; ++c1)
            {
                for(int y = 0; y < wei.mDesc.GetLengths()[2]; ++y)
                {
                    int hi = ho * conv_strides[I0] + y * conv_dilations[I0] - in_left_pads[I0];
                    for(int x = 0; x < wei.mDesc.GetLengths()[3]; ++x)
                    {
                        int wi = wo * conv_strides[I1] + x * conv_dilations[I1] - in_left_pads[I1];
                        if(hi >= 0 && hi < in.mDesc.GetLengths()[2] && wi >= 0 &&
                           wi < in.mDesc.GetLengths()[3])
                        {
                            v += static_cast<const double>(in(n, c0, hi, wi, c1)) *
                                 static_cast<const double>(
                                     wei(k0 * out_host.mDesc.GetLengths()[4] + k1, c0, y, x, c1));
                        }
                    }
                }
            }
        }

        v += bias(k0, k1);
        v = activ(v, activ_type);

        out_host(n, k0, ho, wo, k1) = v;
    };

    make_ParallelTensorFunctor(f_nchw,
                               out_host.mDesc.GetLengths()[0],
                               out_host.mDesc.GetLengths()[1],
                               out_host.mDesc.GetLengths()[2],
                               out_host.mDesc.GetLengths()[3],
                               out_host.mDesc.GetLengths()[4])(std::thread::hardware_concurrency());

    auto maxpool_nchw = [&](auto n, auto k0, auto ho, auto wo, auto k1) {
        auto hx = ho * 2;
        auto wx = wo * 2;

        auto v0 = out_host(n, k0, hx, wx, k1);
        auto v1 = out_host(n, k0, hx, wx + 1, k1);
        auto v2 = out_host(n, k0, hx + 1, wx, k1);
        auto v3 = out_host(n, k0, hx + 1, wx + 1, k1);

        max_host(n, k0, ho, wo, k1) = std::max({v0, v1, v2, v3});
    };

    make_ParallelTensorFunctor(maxpool_nchw,
                               max_host.mDesc.GetLengths()[0],
                               max_host.mDesc.GetLengths()[1],
                               max_host.mDesc.GetLengths()[2],
                               max_host.mDesc.GetLengths()[3],
                               max_host.mDesc.GetLengths()[4])(std::thread::hardware_concurrency());
}

template <typename TIn, typename TWei, typename TOut, typename InLeftPads, typename InRightPads>
void host_winograd_3x3_convolution(const Tensor<TIn>& in_nchw,
                                   const Tensor<TWei>& wei_kcyx,
                                   Tensor<TOut>& out_nkhw,
                                   InLeftPads,
                                   InRightPads)
{
    using namespace ck;

    constexpr std::size_t HoPerTile = 2;
    constexpr std::size_t WoPerTile = 2;

    std::size_t N = in_nchw.mDesc.GetLengths()[0];
    std::size_t C = in_nchw.mDesc.GetLengths()[1];

    std::size_t K = wei_kcyx.mDesc.GetLengths()[0];
    std::size_t Y = wei_kcyx.mDesc.GetLengths()[2];
    std::size_t X = wei_kcyx.mDesc.GetLengths()[3];

    std::size_t Ho = out_nkhw.mDesc.GetLengths()[2];
    std::size_t Wo = out_nkhw.mDesc.GetLengths()[3];

    index_t h_pad_low = InLeftPads{}.Get(Number<0>{});
    index_t w_pad_low = InLeftPads{}.Get(Number<1>{});

    std::size_t HiPerTile = HoPerTile + Y - 1;
    std::size_t WiPerTile = WoPerTile + X - 1;

    std::size_t HTile = (Ho + HoPerTile - 1) / HoPerTile;
    std::size_t WTile = (Wo + WoPerTile - 1) / WoPerTile;

    Tensor<double> in_hold({N, C, HTile, WTile, HiPerTile, WiPerTile});
    Tensor<double> in_transform({N, C, HTile, WTile, HiPerTile, WiPerTile});
    Tensor<double> wei_transform({K, C, HiPerTile, WiPerTile});
    Tensor<double> out_transform({N, K, HTile, WTile, HiPerTile, HiPerTile});
    Tensor<double> out_hold({N, K, HTile, WTile, HoPerTile, WoPerTile});

    auto f_in_hold = [&](auto n, auto c, auto htile, auto wtile) {
        for(int j = 0; j < HiPerTile; ++j)
        {
            int hi = HoPerTile * htile + j - h_pad_low;
            for(int i = 0; i < WiPerTile; ++i)
            {
                int wi = WoPerTile * wtile + i - w_pad_low;

                if(hi >= 0 && hi < in_nchw.mDesc.GetLengths()[2] && wi >= 0 &&
                   wi < in_nchw.mDesc.GetLengths()[3])
                {
                    in_hold(n, c, htile, wtile, j, i) = in_nchw(n, c, hi, wi);
                }
                else
                {
                    in_hold(n, c, htile, wtile, j, i) = TIn(0);
                }
            }
        }
    };

    auto f_in_transform = [&](auto n, auto c, auto htile, auto wtile) {
        in_transform(n, c, htile, wtile, 0, 0) =
            in_hold(n, c, htile, wtile, 0, 0) - in_hold(n, c, htile, wtile, 0, 2) -
            in_hold(n, c, htile, wtile, 2, 0) + in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 0, 1) =
            in_hold(n, c, htile, wtile, 0, 1) + in_hold(n, c, htile, wtile, 0, 2) -
            in_hold(n, c, htile, wtile, 2, 1) - in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 0, 2) =
            -in_hold(n, c, htile, wtile, 0, 1) + in_hold(n, c, htile, wtile, 0, 2) +
            in_hold(n, c, htile, wtile, 2, 1) - in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 0, 3) =
            in_hold(n, c, htile, wtile, 0, 1) - in_hold(n, c, htile, wtile, 0, 3) -
            in_hold(n, c, htile, wtile, 2, 1) + in_hold(n, c, htile, wtile, 2, 3);

        in_transform(n, c, htile, wtile, 1, 0) =
            in_hold(n, c, htile, wtile, 1, 0) - in_hold(n, c, htile, wtile, 1, 2) +
            in_hold(n, c, htile, wtile, 2, 0) - in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 1, 1) =
            in_hold(n, c, htile, wtile, 1, 1) + in_hold(n, c, htile, wtile, 1, 2) +
            in_hold(n, c, htile, wtile, 2, 1) + in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 1, 2) =
            -in_hold(n, c, htile, wtile, 1, 1) + in_hold(n, c, htile, wtile, 1, 2) -
            in_hold(n, c, htile, wtile, 2, 1) + in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 1, 3) =
            in_hold(n, c, htile, wtile, 1, 1) - in_hold(n, c, htile, wtile, 1, 3) +
            in_hold(n, c, htile, wtile, 2, 1) - in_hold(n, c, htile, wtile, 2, 3);

        in_transform(n, c, htile, wtile, 2, 0) =
            -in_hold(n, c, htile, wtile, 1, 0) + in_hold(n, c, htile, wtile, 1, 2) +
            in_hold(n, c, htile, wtile, 2, 0) - in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 2, 1) =
            -in_hold(n, c, htile, wtile, 1, 1) - in_hold(n, c, htile, wtile, 1, 2) +
            in_hold(n, c, htile, wtile, 2, 1) + in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 2, 2) =
            in_hold(n, c, htile, wtile, 1, 1) - in_hold(n, c, htile, wtile, 1, 2) -
            in_hold(n, c, htile, wtile, 2, 1) + in_hold(n, c, htile, wtile, 2, 2);
        in_transform(n, c, htile, wtile, 2, 3) =
            -in_hold(n, c, htile, wtile, 1, 1) + in_hold(n, c, htile, wtile, 1, 3) +
            in_hold(n, c, htile, wtile, 2, 1) - in_hold(n, c, htile, wtile, 2, 3);

        in_transform(n, c, htile, wtile, 3, 0) =
            in_hold(n, c, htile, wtile, 1, 0) - in_hold(n, c, htile, wtile, 1, 2) -
            in_hold(n, c, htile, wtile, 3, 0) + in_hold(n, c, htile, wtile, 3, 2);
        in_transform(n, c, htile, wtile, 3, 1) =
            in_hold(n, c, htile, wtile, 1, 1) + in_hold(n, c, htile, wtile, 1, 2) -
            in_hold(n, c, htile, wtile, 3, 1) - in_hold(n, c, htile, wtile, 3, 2);
        in_transform(n, c, htile, wtile, 3, 2) =
            -in_hold(n, c, htile, wtile, 1, 1) + in_hold(n, c, htile, wtile, 1, 2) +
            in_hold(n, c, htile, wtile, 3, 1) - in_hold(n, c, htile, wtile, 3, 2);
        in_transform(n, c, htile, wtile, 3, 3) =
            in_hold(n, c, htile, wtile, 1, 1) - in_hold(n, c, htile, wtile, 1, 3) -
            in_hold(n, c, htile, wtile, 3, 1) + in_hold(n, c, htile, wtile, 3, 3);
    };

    auto f_wei_transform = [&](auto k, auto c) {
        wei_transform(k, c, 0, 0) = double(wei_kcyx(k, c, 0, 0));
        wei_transform(k, c, 0, 1) = 0.5 * double(wei_kcyx(k, c, 0, 0)) +
                                    0.5 * double(wei_kcyx(k, c, 0, 1)) +
                                    0.5 * double(wei_kcyx(k, c, 0, 2));
        wei_transform(k, c, 0, 2) = 0.5 * double(wei_kcyx(k, c, 0, 0)) -
                                    0.5 * double(wei_kcyx(k, c, 0, 1)) +
                                    0.5 * double(wei_kcyx(k, c, 0, 2));
        wei_transform(k, c, 0, 3) = double(wei_kcyx(k, c, 0, 2));

        wei_transform(k, c, 1, 0) = 0.5 * double(wei_kcyx(k, c, 0, 0)) +
                                    0.5 * double(wei_kcyx(k, c, 1, 0)) +
                                    0.5 * double(wei_kcyx(k, c, 2, 0));
        wei_transform(k, c, 1, 1) =
            0.25 * double(wei_kcyx(k, c, 0, 0)) + 0.25 * double(wei_kcyx(k, c, 0, 1)) +
            0.25 * double(wei_kcyx(k, c, 0, 2)) + 0.25 * double(wei_kcyx(k, c, 1, 0)) +
            0.25 * double(wei_kcyx(k, c, 1, 1)) + 0.25 * double(wei_kcyx(k, c, 1, 2)) +
            0.25 * double(wei_kcyx(k, c, 2, 0)) + 0.25 * double(wei_kcyx(k, c, 2, 1)) +
            0.25 * double(wei_kcyx(k, c, 2, 2));
        wei_transform(k, c, 1, 2) =
            0.25 * double(wei_kcyx(k, c, 0, 0)) - 0.25 * double(wei_kcyx(k, c, 0, 1)) +
            0.25 * double(wei_kcyx(k, c, 0, 2)) + 0.25 * double(wei_kcyx(k, c, 1, 0)) -
            0.25 * double(wei_kcyx(k, c, 1, 1)) + 0.25 * double(wei_kcyx(k, c, 1, 2)) +
            0.25 * double(wei_kcyx(k, c, 2, 0)) - 0.25 * double(wei_kcyx(k, c, 2, 1)) +
            0.25 * double(wei_kcyx(k, c, 2, 2));
        wei_transform(k, c, 1, 3) = 0.5 * double(wei_kcyx(k, c, 0, 2)) +
                                    0.5 * double(wei_kcyx(k, c, 1, 2)) +
                                    0.5 * double(wei_kcyx(k, c, 2, 2));

        wei_transform(k, c, 2, 0) = 0.5 * double(wei_kcyx(k, c, 0, 0)) -
                                    0.5 * double(wei_kcyx(k, c, 1, 0)) +
                                    0.5 * double(wei_kcyx(k, c, 2, 0));
        wei_transform(k, c, 2, 1) =
            0.25 * double(wei_kcyx(k, c, 0, 0)) + 0.25 * double(wei_kcyx(k, c, 0, 1)) +
            0.25 * double(wei_kcyx(k, c, 0, 2)) - 0.25 * double(wei_kcyx(k, c, 1, 0)) -
            0.25 * double(wei_kcyx(k, c, 1, 1)) - 0.25 * double(wei_kcyx(k, c, 1, 2)) +
            0.25 * double(wei_kcyx(k, c, 2, 0)) + 0.25 * double(wei_kcyx(k, c, 2, 1)) +
            0.25 * double(wei_kcyx(k, c, 2, 2));
        wei_transform(k, c, 2, 2) =
            0.25 * double(wei_kcyx(k, c, 0, 0)) - 0.25 * double(wei_kcyx(k, c, 0, 1)) +
            0.25 * double(wei_kcyx(k, c, 0, 2)) - 0.25 * double(wei_kcyx(k, c, 1, 0)) +
            0.25 * double(wei_kcyx(k, c, 1, 1)) - 0.25 * double(wei_kcyx(k, c, 1, 2)) +
            0.25 * double(wei_kcyx(k, c, 2, 0)) - 0.25 * double(wei_kcyx(k, c, 2, 1)) +
            0.25 * double(wei_kcyx(k, c, 2, 2));
        wei_transform(k, c, 2, 3) = 0.5 * double(wei_kcyx(k, c, 0, 2)) -
                                    0.5 * double(wei_kcyx(k, c, 1, 2)) +
                                    0.5 * double(wei_kcyx(k, c, 2, 2));

        wei_transform(k, c, 3, 0) = double(wei_kcyx(k, c, 2, 0));
        wei_transform(k, c, 3, 1) = 0.5 * double(wei_kcyx(k, c, 2, 0)) +
                                    0.5 * double(wei_kcyx(k, c, 2, 1)) +
                                    0.5 * double(wei_kcyx(k, c, 2, 2));
        wei_transform(k, c, 3, 2) = 0.5 * double(wei_kcyx(k, c, 2, 0)) -
                                    0.5 * double(wei_kcyx(k, c, 2, 1)) +
                                    0.5 * double(wei_kcyx(k, c, 2, 2));
        wei_transform(k, c, 3, 3) = double(wei_kcyx(k, c, 2, 2));
    };

    auto f_out_transform = [&](auto n, auto k, auto htile, auto wtile) {
        for(int j = 0; j < HiPerTile; ++j)
        {
            for(int i = 0; i < WiPerTile; ++i)
            {
                double v = 0;
                for(int c = 0; c < C; ++c)
                {
                    v += in_transform(n, c, htile, wtile, j, i) * wei_transform(k, c, j, i);
                }

                out_transform(n, k, htile, wtile, j, i) = v;
            }
        }
    };

    auto f_out_hold = [&](auto n, auto k, auto htile, auto wtile) {
        out_hold(n, k, htile, wtile, 0, 0) =
            out_transform(n, k, htile, wtile, 0, 0) + out_transform(n, k, htile, wtile, 0, 1) +
            out_transform(n, k, htile, wtile, 0, 2) + out_transform(n, k, htile, wtile, 1, 0) +
            out_transform(n, k, htile, wtile, 1, 1) + out_transform(n, k, htile, wtile, 1, 2) +
            out_transform(n, k, htile, wtile, 2, 0) + out_transform(n, k, htile, wtile, 2, 1) +
            out_transform(n, k, htile, wtile, 2, 2);
        out_hold(n, k, htile, wtile, 0, 1) =
            out_transform(n, k, htile, wtile, 0, 1) - out_transform(n, k, htile, wtile, 0, 2) -
            out_transform(n, k, htile, wtile, 0, 3) + out_transform(n, k, htile, wtile, 1, 1) -
            out_transform(n, k, htile, wtile, 1, 2) - out_transform(n, k, htile, wtile, 1, 3) +
            out_transform(n, k, htile, wtile, 2, 1) - out_transform(n, k, htile, wtile, 2, 2) -
            out_transform(n, k, htile, wtile, 2, 3);
        out_hold(n, k, htile, wtile, 1, 0) =
            out_transform(n, k, htile, wtile, 1, 0) + out_transform(n, k, htile, wtile, 1, 1) +
            out_transform(n, k, htile, wtile, 1, 2) - out_transform(n, k, htile, wtile, 2, 0) -
            out_transform(n, k, htile, wtile, 2, 1) - out_transform(n, k, htile, wtile, 2, 2) -
            out_transform(n, k, htile, wtile, 3, 0) - out_transform(n, k, htile, wtile, 3, 1) -
            out_transform(n, k, htile, wtile, 3, 2);
        out_hold(n, k, htile, wtile, 1, 1) =
            out_transform(n, k, htile, wtile, 1, 1) - out_transform(n, k, htile, wtile, 1, 2) -
            out_transform(n, k, htile, wtile, 1, 3) - out_transform(n, k, htile, wtile, 2, 1) +
            out_transform(n, k, htile, wtile, 2, 2) + out_transform(n, k, htile, wtile, 2, 3) -
            out_transform(n, k, htile, wtile, 3, 1) + out_transform(n, k, htile, wtile, 3, 2) +
            out_transform(n, k, htile, wtile, 3, 3);
    };

    auto f_out = [&](auto n, auto k, auto htile, auto wtile) {
        for(int j = 0; j < HoPerTile; ++j)
        {
            std::size_t ho = HoPerTile * htile + j;
            for(int i = 0; i < WoPerTile; ++i)
            {
                std::size_t wo         = WoPerTile * wtile + i;
                out_nkhw(n, k, ho, wo) = out_hold(n, k, htile, wtile, j, i);
            }
        }
    };

    std::size_t num_thread = std::thread::hardware_concurrency();

    make_ParallelTensorFunctor(f_in_hold, N, C, HTile, WTile)(num_thread);
    make_ParallelTensorFunctor(f_in_transform, N, C, HTile, WTile)(num_thread);
    make_ParallelTensorFunctor(f_wei_transform, K, C)(num_thread);
    make_ParallelTensorFunctor(f_out_transform, N, K, HTile, WTile)(num_thread);
    make_ParallelTensorFunctor(f_out_hold, N, K, HTile, WTile)(num_thread);
    make_ParallelTensorFunctor(f_out, N, K, HTile, WTile)(num_thread);
}
