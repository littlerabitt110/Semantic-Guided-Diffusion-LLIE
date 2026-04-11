import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune  # <-- Added for pruning
import os
import model.networks as networks
from .base_model import BaseModel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.cuda.amp as amp  # <-- AMP Support Added

logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)

        if opt['dist']:
            self.local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        if opt['dist']:
            self.netG.to(device)

        if not opt['uncertainty_train']:
            self.netGU = self.set_device(networks.define_G(opt))  # Uncertainty model
            if opt['dist']:
                self.netGU.to(device)

        self.schedule_phase = None
        self.opt = opt

        # Apply pruning to the model
        self.apply_pruning(pruning_amount=0.3)  # <-- Added pruning

        # Set loss and load resume state
        self.set_loss()  # ✅ Keep `set_loss()`
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train'
        )  # ✅ Keep `set_new_noise_schedule()`

        # AMP GradScaler
        self.use_amp = self.opt['train'].get('use_amp', False)
        self.scaler = amp.GradScaler() if self.use_amp else None

        if self.opt['phase'] == 'train':
            self.netG.train()
            optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"]
            )
            self.log_dict = OrderedDict()

        if not opt['uncertainty_train'] and self.opt['phase'] == 'train':
            self.netGU.load_state_dict(torch.load(self.opt['path']['resume_state'] + '_gen.pth'), strict=True)
            if opt['dist']:
                self.netGU = DDP(self.netGU, device_ids=[self.local_rank],
                                 output_device=self.local_rank, find_unused_parameters=True)

        # Load model if resuming training or testing
        self.load_network()

        if opt['dist']:
            self.netG = DDP(self.netG, device_ids=[self.local_rank],
                            output_device=self.local_rank, find_unused_parameters=True)
        self.print_network()

    # >>> Function to Apply Pruning <<<
    def apply_pruning(self, pruning_amount=0.3):
        """
        Applies pruning to convolutional and linear layers.
        :param pruning_amount: Fraction of weights to remove (default: 30%).
        """
        for name, module in self.netG.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=pruning_amount)
                prune.remove(module, 'weight')  # Remove pruning hooks to make it permanent

        logger.info(f"Applied {pruning_amount * 100}% pruning to the model.")

    def set_loss(self):
        """
        Ensures the model has its loss functions set up correctly.
        """
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

        if not self.opt['uncertainty_train']:
            if isinstance(self.netGU, nn.DataParallel):
                self.netGU.module.set_loss(self.device)
            else:
                self.netGU.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        """
        Ensures the correct noise schedule is used during training/testing.
        """
        if self.opt['dist']:
            device = torch.device("cuda", self.local_rank)
        else:
            device = self.device

        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(schedule_opt, device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, device)

            if not self.opt['uncertainty_train']:
                if isinstance(self.netGU, nn.DataParallel):
                    self.netGU.module.set_new_noise_schedule(schedule_opt, device)
                else:
                    self.netGU.set_new_noise_schedule(schedule_opt, device)

    def get_current_log(self):
        """
        Returns the current log dictionary containing training loss values.
        """
        return self.log_dict

    def load_network(self):
        """
        Loads the pretrained model if resuming training or testing.
        """
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(f'Loading pretrained model for G [{load_path}] ...')
            gen_path = f'{load_path}_gen.pth'

            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module

            network.load_state_dict(torch.load(gen_path), strict=True)

            if self.opt['phase'] == 'train':
                self.begin_step = 0
                self.begin_epoch = 0

    def feed_data(self, data):
        if not hasattr(self, 'data'):
            self.data = {}  # Ensure self.data exists

        if self.opt['dist']:
            self.data['LQ'] = data['LQ'].to(self.local_rank)
            self.data['GT'] = data['GT'].to(self.local_rank)
        else:
            self.data['LQ'] = self.set_device(data['LQ'])
            self.data['GT'] = self.set_device(data['GT'])

    def optimize_parameters(self):
        if not hasattr(self, 'data'):
            raise ValueError("Error: self.data is not set! Call feed_data() before optimize_parameters().")

        self.optG.zero_grad()

        if self.use_amp:
            with amp.autocast():
                loss = self.compute_loss()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optG)
            self.scaler.update()
        else:
            loss = self.compute_loss()
            loss.backward()
            self.optG.step()

        self.log_dict['total_loss'] = loss.item()

    def compute_loss(self):
        """
        Computes the loss function.
        """
        if not self.opt['uncertainty_train']:
            if self.opt['dist']:
                l_pix, l_gsad = self.netG(self.data, self.netGU.module.denoise_fn)
            else:
                l_pix, l_gsad = self.netG(self.data, self.netGU.denoise_fn)

            b, c, h, w = self.data['LQ'].shape
            num_clusters = 6
            l_pix = l_pix.sum() / int(b * c * h * w)
            l_gsad = l_gsad.sum() / int(b * num_clusters)
            loss = l_pix + l_gsad

            self.log_dict['l_1'] = l_pix.item()
            self.log_dict['l_gsad'] = l_gsad.item()
            return loss

        else:
            l_pix = self.netG(self.data)
            b, c, h, w = self.data['LQ'].shape
            loss = l_pix.sum() / int(b * c * h * w)

            self.log_dict['l_u'] = loss.item()
            return loss
